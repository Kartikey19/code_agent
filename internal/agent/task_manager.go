package agent

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// TaskStatus represents the status of a task
type TaskStatus string

const (
	TaskStatusPending    TaskStatus = "pending"
	TaskStatusInProgress TaskStatus = "in_progress"
	TaskStatusCompleted  TaskStatus = "completed"
	TaskStatusBlocked    TaskStatus = "blocked"
)

// Task represents a single task in a breakdown
type Task struct {
	ID          int        `json:"id"`
	Description string     `json:"description"`
	Status      TaskStatus `json:"status"`
	Details     string     `json:"details,omitempty"`
	FilePath    string     `json:"file_path,omitempty"`
	Line        int        `json:"line,omitempty"`
}

// TaskBreakdown represents a complete breakdown of tasks for a user prompt
type TaskBreakdown struct {
	UserPrompt string `json:"user_prompt"`
	Summary    string `json:"summary"`
	Tasks      []Task `json:"tasks"`
	TotalTasks int    `json:"total_tasks"`
	Completed  int    `json:"completed"`
	InProgress int    `json:"in_progress"`
	Pending    int    `json:"pending"`
}

// TaskManager handles task parsing, tracking, and formatting
type TaskManager struct{}

// NewTaskManager creates a new task manager
func NewTaskManager() *TaskManager {
	return &TaskManager{}
}

// ParseTasksFromLLM parses tasks from LLM response
// It looks for common task list formats:
// - ☐ Task description
// - [ ] Task description
// - 1. Task description
// - - Task description
func (tm *TaskManager) ParseTasksFromLLM(llmResponse string) (*TaskBreakdown, error) {
	lines := strings.Split(llmResponse, "\n")
	var tasks []Task
	taskID := 1

	// Regex patterns for different task formats
	checkboxPattern := regexp.MustCompile(`^[\s]*[☐☑✓✗\[\]x\s-]+\s*(.+)$`)
	numberedPattern := regexp.MustCompile(`^[\s]*\d+\.\s+(.+)$`)
	bulletPattern := regexp.MustCompile(`^[\s]*[-*•]\s+(.+)$`)

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		var description string
		var status TaskStatus = TaskStatusPending

		// Check for completed tasks (☑, ✓, [x])
		if strings.Contains(line, "☑") || strings.Contains(line, "✓") || strings.Contains(line, "[x]") {
			status = TaskStatusCompleted
		}

		// Try checkbox format
		if matches := checkboxPattern.FindStringSubmatch(line); len(matches) > 1 {
			description = strings.TrimSpace(matches[1])
		} else if matches := numberedPattern.FindStringSubmatch(line); len(matches) > 1 {
			// Try numbered format
			description = strings.TrimSpace(matches[1])
		} else if matches := bulletPattern.FindStringSubmatch(line); len(matches) > 1 {
			// Try bullet format
			description = strings.TrimSpace(matches[1])
		} else {
			// Skip lines that don't match task patterns
			continue
		}

		if description != "" {
			tasks = append(tasks, Task{
				ID:          taskID,
				Description: description,
				Status:      status,
			})
			taskID++
		}
	}

	if len(tasks) == 0 {
		return nil, fmt.Errorf("no tasks found in LLM response")
	}

	breakdown := &TaskBreakdown{
		Tasks:      tasks,
		TotalTasks: len(tasks),
	}

	breakdown.UpdateStats()
	return breakdown, nil
}

// CreateTaskBreakdown creates a task breakdown with statistics
func (tm *TaskManager) CreateTaskBreakdown(userPrompt string, tasks []Task) *TaskBreakdown {
	breakdown := &TaskBreakdown{
		UserPrompt: userPrompt,
		Tasks:      tasks,
		TotalTasks: len(tasks),
	}

	breakdown.UpdateStats()
	return breakdown
}

// UpdateStats updates the statistics in a task breakdown
func (tb *TaskBreakdown) UpdateStats() {
	tb.Completed = 0
	tb.InProgress = 0
	tb.Pending = 0

	for _, task := range tb.Tasks {
		switch task.Status {
		case TaskStatusCompleted:
			tb.Completed++
		case TaskStatusInProgress:
			tb.InProgress++
		case TaskStatusPending:
			tb.Pending++
		}
	}
}

// UpdateTaskStatus updates the status of a task
func (tb *TaskBreakdown) UpdateTaskStatus(taskID int, status TaskStatus) error {
	for i, task := range tb.Tasks {
		if task.ID == taskID {
			tb.Tasks[i].Status = status
			tb.UpdateStats()
			return nil
		}
	}
	return fmt.Errorf("task with ID %d not found", taskID)
}

// FormatAsChecklist formats tasks as a checkbox list
func (tm *TaskManager) FormatAsChecklist(breakdown *TaskBreakdown) string {
	var b strings.Builder

	if breakdown.Summary != "" {
		b.WriteString(fmt.Sprintf("# %s\n\n", breakdown.Summary))
	}

	b.WriteString(fmt.Sprintf("**Progress:** %d/%d tasks completed\n\n",
		breakdown.Completed, breakdown.TotalTasks))

	for _, task := range breakdown.Tasks {
		checkbox := "☐"
		switch task.Status {
		case TaskStatusCompleted:
			checkbox = "☑"
		case TaskStatusInProgress:
			checkbox = "◐"
		}

		b.WriteString(fmt.Sprintf("%s %s", checkbox, task.Description))
		if task.FilePath != "" {
			b.WriteString(fmt.Sprintf(" (%s", task.FilePath))
			if task.Line > 0 {
				b.WriteString(fmt.Sprintf(":%d", task.Line))
			}
			b.WriteString(")")
		}
		b.WriteString("\n")

		if task.Details != "" {
			// Indent details
			details := strings.ReplaceAll(task.Details, "\n", "\n  ")
			b.WriteString(fmt.Sprintf("  %s\n", details))
		}
	}

	return b.String()
}

// FormatAsJSON formats the task breakdown as JSON
func (tm *TaskManager) FormatAsJSON(breakdown *TaskBreakdown) (string, error) {
	data, err := json.MarshalIndent(breakdown, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// GenerateTaskPrompt generates a prompt for the LLM to create a task breakdown
func (tm *TaskManager) GenerateTaskPrompt(userPrompt, projectContext string) string {
	return fmt.Sprintf(`You are a coding agent task planner. Given a user's request and project context, create a detailed task breakdown.

USER REQUEST:
%s

PROJECT CONTEXT:
%s

Please create a detailed task breakdown in the following format:
☐ Task 1 description
☐ Task 2 description
☐ Task 3 description
...

IMPORTANT:
- Each task should be specific and actionable
- Include file paths when relevant (e.g., "Check schemas/patient.py for field definitions")
- Order tasks logically (investigation → implementation → testing)
- Be concise but clear
- Use checkbox format (☐) for pending tasks
- Focus on the most critical tasks first

Your task breakdown:`, userPrompt, projectContext)
}
