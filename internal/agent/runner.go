package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/yourorg/agent/internal/indexer"
)

// RunOptions controls the autonomous execution loop.
type RunOptions struct {
	DryRun            bool
	MaxIterations     int
	MaxContextResults int
}

// RunResult is returned after running the full agent loop.
type RunResult struct {
	Plan       *TaskBreakdown  `json:"plan"`
	Executions []TaskExecution `json:"executions"`
}

// Run executes the full agent loop: plan → execute tasks → report.
func (a *CodingAgent) Run(ctx context.Context, userPrompt string, opts RunOptions) (*RunResult, error) {
	if opts.MaxIterations <= 0 {
		opts.MaxIterations = 25
	}
	if opts.MaxContextResults <= 0 {
		opts.MaxContextResults = 8
	}

	// Build or load project index once for the session.
	projectIndex, err := a.indexer.IndexProject(a.projectPath)
	if err != nil {
		return nil, fmt.Errorf("failed to index project: %w", err)
	}

	// Generate plan up front.
	plan, err := a.PlanTask(ctx, userPrompt)
	if err != nil {
		return nil, err
	}

	executor := NewExecutor(ExecutorConfig{
		ProjectRoot: a.projectPath,
		Index:       projectIndex,
		DryRun:      opts.DryRun,
	})

	contextFetcher := indexer.NewContextFetcher(projectIndex)
	var executions []TaskExecution

	for i, task := range plan.Tasks {
		_ = plan.UpdateTaskStatus(task.ID, TaskStatusInProgress)

		taskContext := contextFetcher.FetchContext(task.Description, opts.MaxContextResults)
		contextString := indexer.FormatContext(taskContext)

		execResult := a.executeTask(ctx, executor, task, contextString, opts.MaxIterations)
		executions = append(executions, execResult)

		switch {
		case execResult.Completed:
			_ = plan.UpdateTaskStatus(task.ID, TaskStatusCompleted)
		case execResult.Failed:
			_ = plan.UpdateTaskStatus(task.ID, TaskStatusBlocked)
		default:
			_ = plan.UpdateTaskStatus(task.ID, TaskStatusPending)
		}

		plan.Tasks[i].Details = fmt.Sprintf("Ran %d action(s)", len(execResult.Actions))
	}

	plan.UpdateStats()

	return &RunResult{
		Plan:       plan,
		Executions: executions,
	}, nil
}

func (a *CodingAgent) executeTask(ctx context.Context, executor *Executor, task Task, contextString string, maxIterations int) TaskExecution {
	var (
		actions []Action
		results []ActionResult
	)

	history := make([]string, 0, maxIterations)

	for i := 0; i < maxIterations; i++ {
		prompt := buildActionDecisionPrompt(task.Description, contextString, history)

		response, err := a.llmClient.Chat(ctx, []Message{
			{Role: "system", Content: "You are executing a coding task. Pick and emit ONE action in JSON. Do not add commentary outside JSON."},
			{Role: "user", Content: prompt},
		})
		if err != nil {
			return TaskExecution{Task: task, Failed: true, FailureMsg: fmt.Sprintf("llm error: %v", err)}
		}

		var action Action
		if err := json.Unmarshal([]byte(strings.TrimSpace(response.Content)), &action); err != nil {
			return TaskExecution{Task: task, Failed: true, FailureMsg: fmt.Sprintf("could not parse action JSON: %v", err)}
		}

		actions = append(actions, action)
		result := executor.Execute(ctx, action)
		results = append(results, result)

		// Append brief history for the next iteration
		history = append(history, summarizeStep(action, result))

		if action.Type == ActionComplete || action.Type == ActionFail {
			return TaskExecution{
				Task:       task,
				Actions:    actions,
				Results:    results,
				Completed:  action.Type == ActionComplete && result.Success,
				Failed:     action.Type == ActionFail || !result.Success,
				FailureMsg: result.Error,
			}
		}

		// If execution failed hard, surface it
		if !result.Success {
			return TaskExecution{
				Task:       task,
				Actions:    actions,
				Results:    results,
				Failed:     true,
				FailureMsg: result.Error,
			}
		}
	}

	return TaskExecution{
		Task:       task,
		Actions:    actions,
		Results:    results,
		Failed:     true,
		FailureMsg: "max iterations reached before completion",
	}
}

func buildActionDecisionPrompt(taskDesc, contextString string, history []string) string {
	var b strings.Builder

	b.WriteString("CURRENT TASK:\n")
	b.WriteString(taskDesc)
	b.WriteString("\n\nPROJECT CONTEXT:\n")
	b.WriteString(contextString)

	if len(history) > 0 {
		b.WriteString("\n\nPREVIOUS STEPS:\n")
		for _, h := range history {
			b.WriteString("- ")
			b.WriteString(h)
			b.WriteString("\n")
		}
	}

	b.WriteString(`

You can take exactly ONE of these actions:
- read_file: { "type": "read_file", "path": "<relative path>" }
- edit_file: { "type": "edit_file", "path": "<relative path>", "edits": [{ "old_text": "...", "new_text": "..." }] }
- create_file: { "type": "create_file", "path": "<relative path>", "content": "full file content" }
- delete_file: { "type": "delete_file", "path": "<relative path>" }
- run_command: { "type": "run_command", "command": "<shell command>", "workdir": "<dir>", "timeout": 120 }
- search: { "type": "search", "query": "<symbol or keyword>" }
- ask_user: { "type": "ask_user", "question": "<clarifying question>" }
- complete: { "type": "complete", "summary": "what you accomplished" }
- fail: { "type": "fail", "reason": "why you cannot proceed" }

Respond with a single JSON object describing the action.`)

	return b.String()
}

func summarizeStep(action Action, result ActionResult) string {
	var status string
	if result.Success {
		status = "ok"
	} else {
		status = "err"
	}

	output := strings.TrimSpace(result.Output)
	if len(output) > 240 {
		output = output[:240] + "..."
	}

	return fmt.Sprintf("%s %s → %s (%s)", action.Type, action.Path, status, output)
}
