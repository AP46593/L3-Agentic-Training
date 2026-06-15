# Project Structure

```
/                             # Workspace root
├── .kiro/                    # Kiro configuration & steering
│   └── steering/             # AI assistant guidance docs
├── .venv/                    # Python virtual environment (gitignored)
├── L3-Agentic-Training/      # Main project (git repo)
│   ├── requirements.txt      # Python dependencies
│   ├── .gitignore
│   └── .gitattributes
└── readme.txt                # Quick-start setup instructions
```

## Conventions
- The workspace root contains the virtual environment and the main project repo as a subdirectory.
- Source code, notebooks, and training materials live inside `L3-Agentic-Training/`.
- Dependencies are pinned with minimum versions in `requirements.txt`.
- Environment secrets go in `.env` at the workspace or project root (never committed).
- Jupyter notebooks and Python scripts are the primary file types.

## When Adding New Files
- Place new training modules, notebooks, or scripts inside `L3-Agentic-Training/`.
- Keep the virtual environment at the workspace root (`.venv/`).
- If adding new dependencies, append them to `L3-Agentic-Training/requirements.txt` with a minimum version pin.
