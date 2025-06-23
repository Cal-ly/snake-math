# CodeFold Normalizer

A comprehensive Python script for managing code block wrapping in Markdown files. This tool ensures that all code blocks are consistently wrapped with `<CodeFold>` tags for better presentation and maintainability.

## Features

- **Check**: Identify unwrapped code blocks in Markdown files
- **Fix**: Automatically wrap code blocks with proper `<CodeFold>` tags
- **Verify**: Validate that all code blocks follow the correct formatting
- **All**: Run the complete normalization process (check → fix → verify)

## Usage

```bash
# Check for unwrapped code blocks
python codefold_normalizer.py check [directory]

# Fix unwrapped code blocks
python codefold_normalizer.py fix [directory]

# Verify proper formatting
python codefold_normalizer.py verify [directory]

# Run complete process (check, fix, verify)
python codefold_normalizer.py all [directory]
```

If no directory is specified, the script defaults to `docs/` in the current working directory.

## Expected Format

The script ensures all code blocks follow this format:

```markdown
<CodeFold>

```language
code content here
```

</CodeFold>
```

## Examples

```bash
# Check the docs directory for issues
python codefold_normalizer.py check

# Fix all code blocks in a specific directory
python codefold_normalizer.py fix content/

# Verify formatting after making changes
python codefold_normalizer.py verify docs/

# Run the complete normalization process
python codefold_normalizer.py all
```

## Requirements

- Python 3.6+
- No external dependencies (uses only standard library)

## What It Does

1. **Identifies** code blocks (delimited by triple backticks)
2. **Wraps** them with `<CodeFold>` tags if not already wrapped
3. **Ensures** proper spacing (empty lines around code blocks)
4. **Preserves** all original code content and language specifications
5. **Removes** any duplicate or malformed wrapping

This tool is particularly useful for documentation projects that use custom components like `<CodeFold>` for enhanced code presentation.
