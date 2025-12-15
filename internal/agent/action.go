package agent

import "time"

// ActionType represents the kind of operation the agent can perform.
type ActionType string

const (
	ActionReadFile   ActionType = "read_file"
	ActionEditFile   ActionType = "edit_file"
	ActionCreateFile ActionType = "create_file"
	ActionDeleteFile ActionType = "delete_file"
	ActionRunCommand ActionType = "run_command"
	ActionSearch     ActionType = "search"
	ActionAskUser    ActionType = "ask_user"
	ActionComplete   ActionType = "complete"
	ActionFail       ActionType = "fail"
)

// TextEdit represents a search/replace operation within a file.
type TextEdit struct {
	OldText string `json:"old_text"`
	NewText string `json:"new_text"`
}

// Action is a single instruction emitted by the LLM.
type Action struct {
	Type     ActionType `json:"type"`
	Path     string     `json:"path,omitempty"`
	Edits    []TextEdit `json:"edits,omitempty"`
	Content  string     `json:"content,omitempty"`
	Command  string     `json:"command,omitempty"`
	Workdir  string     `json:"workdir,omitempty"`
	Query    string     `json:"query,omitempty"`
	Reason   string     `json:"reason,omitempty"`
	Timeout  int        `json:"timeout,omitempty"` // seconds
	Summary  string     `json:"summary,omitempty"`
	Question string     `json:"question,omitempty"`
}

// ActionResult captures the outcome of executing an action.
type ActionResult struct {
	Success      bool          `json:"success"`
	Output       string        `json:"output,omitempty"`
	Error        string        `json:"error,omitempty"`
	FilesChanged []string      `json:"files_changed,omitempty"`
	Duration     time.Duration `json:"duration,omitempty"`
}

// TaskExecution contains the record of a single task's execution loop.
type TaskExecution struct {
	Task       Task           `json:"task"`
	Actions    []Action       `json:"actions,omitempty"`
	Results    []ActionResult `json:"results,omitempty"`
	Completed  bool           `json:"completed"`
	Failed     bool           `json:"failed"`
	FailureMsg string         `json:"failure_msg,omitempty"`
}
