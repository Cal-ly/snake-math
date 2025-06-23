#!/usr/bin/env python3
"""
CodeFold Normalizer - A comprehensive tool for managing code block wrapping in Markdown files.

This script provides functionality to:
1. Check for unwrapped code blocks
2. Fix unwrapped code blocks by adding <CodeFold> tags
3. Verify that all code blocks are properly formatted

Usage:
    python codefold_normalizer.py check [directory]    - Check for unwrapped code blocks
    python codefold_normalizer.py fix [directory]      - Fix unwrapped code blocks
    python codefold_normalizer.py verify [directory]   - Verify proper formatting
    python codefold_normalizer.py all [directory]      - Run all operations (check, fix, verify)

If no directory is specified, defaults to 'docs/' in the current working directory.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict


class CodeFoldNormalizer:
    """Main class for managing code block normalization in Markdown files."""
    
    def __init__(self, docs_dir: str = "docs"):
        """Initialize with the documentation directory path."""
        self.docs_dir = Path(docs_dir)
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Directory '{docs_dir}' does not exist")
    
    def find_unwrapped_code_blocks(self, file_path: Path) -> List[Tuple[int, str]]:
        """
        Find code blocks that are NOT wrapped in <CodeFold> tags.
        
        Args:
            file_path: Path to the markdown file to check
            
        Returns:
            List of tuples containing (line_number, code_block_start_line)
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        unwrapped_blocks = []
        
        # Find all code blocks with their positions
        pattern = r'(```\w*\n.*?\n```)'
        
        for match in re.finditer(pattern, content, re.DOTALL):
            start_pos = match.start()
            end_pos = match.end()
            code_block = match.group(1)
            
            # Find the text before and after this code block
            before_text = content[:start_pos]
            after_text = content[end_pos:]
            
            # Check if there's a <CodeFold> before and </CodeFold> after
            has_opening_tag = '<CodeFold>' in before_text[-200:]  # Check last 200 chars before
            has_closing_tag = '</CodeFold>' in after_text[:200]  # Check first 200 chars after
            
            if not (has_opening_tag and has_closing_tag):
                # Count line number of the start
                line_num = content[:start_pos].count('\n') + 1
                unwrapped_blocks.append((line_num, code_block.split('\n')[0]))
        
        return unwrapped_blocks
    
    def fix_code_blocks(self, file_path: Path) -> bool:
        """
        Fix unwrapped code blocks by adding proper <CodeFold> tags.
        
        Args:
            file_path: Path to the markdown file to fix
            
        Returns:
            True if file was modified, False otherwise
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Step 1: Remove all existing CodeFold tags to start fresh
        content = re.sub(r'<CodeFold>\s*\n\s*\n', '', content)
        content = re.sub(r'\n\s*\n\s*</CodeFold>', '', content)
        content = re.sub(r'<CodeFold>', '', content)
        content = re.sub(r'</CodeFold>', '', content)
        
        # Step 2: Find all code blocks and wrap them properly
        def wrap_code_block(match):
            code_block = match.group(0)
            return f'<CodeFold>\n\n{code_block}\n\n</CodeFold>'
        
        # Pattern to match complete code blocks (```lang...```)
        pattern = r'```\w*\n.*?\n```'
        content = re.sub(pattern, wrap_code_block, content, flags=re.DOTALL)
        
        # Step 3: Clean up excessive whitespace
        content = re.sub(r'\n\n\n+', '\n\n', content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
    
    def verify_code_block_format(self, file_path: Path) -> Tuple[List[str], int, int]:
        """
        Verify that all code blocks follow the correct format.
        
        Args:
            file_path: Path to the markdown file to verify
            
        Returns:
            Tuple of (issues_list, wrapped_count, total_count)
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        issues = []
        
        # Find all <CodeFold> blocks
        pattern = r'<CodeFold>\s*\n\s*\n\s*(```\w*\n.*?\n```)\s*\n\s*\n\s*</CodeFold>'
        matches = list(re.finditer(pattern, content, re.DOTALL))
        
        # Also find any code blocks not in CodeFold
        all_code_blocks = list(re.finditer(r'```\w*\n.*?\n```', content, re.DOTALL))
        
        wrapped_code_blocks = len(matches)
        total_code_blocks = len(all_code_blocks)
        
        if wrapped_code_blocks != total_code_blocks:
            issues.append(f"Found {total_code_blocks} code blocks but only {wrapped_code_blocks} are properly wrapped")
        
        # Check format of wrapped blocks
        for i, match in enumerate(matches):
            full_match = match.group(0)
            
            # Check if there's proper spacing
            if not re.match(r'<CodeFold>\s*\n\s*\n', full_match):
                issues.append(f"Code block {i+1}: Missing empty line after <CodeFold>")
            
            if not re.search(r'\n\s*\n\s*</CodeFold>$', full_match):
                issues.append(f"Code block {i+1}: Missing empty line before </CodeFold>")
        
        return issues, wrapped_code_blocks, total_code_blocks
    
    def check_all_files(self) -> Dict[str, List[Tuple[int, str]]]:
        """
        Check all markdown files for unwrapped code blocks.
        
        Returns:
            Dictionary mapping file paths to lists of unwrapped blocks
        """
        results = {}
        
        for md_file in self.docs_dir.glob('**/*.md'):
            unwrapped = self.find_unwrapped_code_blocks(md_file)
            if unwrapped:
                results[str(md_file.relative_to(self.docs_dir))] = unwrapped
        
        return results
    
    def fix_all_files(self) -> int:
        """
        Fix code blocks in all markdown files.
        
        Returns:
            Number of files that were modified
        """
        files_modified = 0
        
        for md_file in self.docs_dir.glob('**/*.md'):
            if self.fix_code_blocks(md_file):
                files_modified += 1
        
        return files_modified
    
    def verify_all_files(self) -> Tuple[Dict[str, List[str]], int, int, int]:
        """
        Verify formatting of all markdown files.
        
        Returns:
            Tuple of (issues_dict, files_with_issues, total_wrapped, total_blocks)
        """
        all_issues = {}
        files_with_issues = 0
        total_wrapped = 0
        total_blocks = 0
        
        for md_file in self.docs_dir.glob('**/*.md'):
            issues, wrapped, total = self.verify_code_block_format(md_file)
            total_wrapped += wrapped
            total_blocks += total
            
            if issues:
                all_issues[str(md_file.relative_to(self.docs_dir))] = issues
                files_with_issues += 1
        
        return all_issues, files_with_issues, total_wrapped, total_blocks


def print_check_results(results: Dict[str, List[Tuple[int, str]]]):
    """Print the results of checking for unwrapped code blocks."""
    print("Checking for unwrapped code blocks...")
    print("=" * 50)
    
    if not results:
        print("\n✅ All code blocks are properly wrapped with <CodeFold> tags!")
        return
    
    total_unwrapped = 0
    for file_path, unwrapped_blocks in results.items():
        print(f"\n{file_path}:")
        for line_num, code_line in unwrapped_blocks:
            print(f"  Line {line_num}: {code_line}")
            total_unwrapped += 1
    
    print(f"\n❌ Total unwrapped code blocks found: {total_unwrapped}")


def print_fix_results(files_modified: int):
    """Print the results of fixing code blocks."""
    print("Fixing unwrapped code blocks...")
    print("=" * 40)
    
    if files_modified == 0:
        print("\n✅ No files needed modification - all code blocks were already properly wrapped!")
    else:
        print(f"\n✅ Successfully fixed code blocks in {files_modified} files!")


def print_verify_results(issues_dict: Dict[str, List[str]], files_with_issues: int, 
                        total_wrapped: int, total_blocks: int):
    """Print the results of verification."""
    print("Verifying code block formatting...")
    print("=" * 45)
    
    if issues_dict:
        for file_path, issues in issues_dict.items():
            print(f"\n❌ {file_path}:")
            for issue in issues:
                print(f"   {issue}")
    
    print(f"\n" + "=" * 45)
    print(f"Summary:")
    print(f"  Files with issues: {files_with_issues}")
    print(f"  Total code blocks: {total_blocks}")
    print(f"  Properly wrapped: {total_wrapped}")
    
    if files_with_issues == 0 and total_wrapped == total_blocks:
        print(f"\n🎉 SUCCESS! All {total_blocks} code blocks are properly formatted!")
    else:
        print(f"\n⚠️  Some issues found. Review the output above.")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="CodeFold Normalizer - Manage code block wrapping in Markdown files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python codefold_normalizer.py check           # Check docs/ directory
  python codefold_normalizer.py fix docs/       # Fix code blocks in docs/
  python codefold_normalizer.py verify          # Verify formatting
  python codefold_normalizer.py all content/    # Run all operations on content/
        """
    )
    
    parser.add_argument(
        'operation',
        choices=['check', 'fix', 'verify', 'all'],
        help='Operation to perform'
    )
    
    parser.add_argument(
        'directory',
        nargs='?',
        default='docs',
        help='Directory containing markdown files (default: docs)'
    )
    
    args = parser.parse_args()
    
    try:
        normalizer = CodeFoldNormalizer(args.directory)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    if args.operation == 'check':
        results = normalizer.check_all_files()
        print_check_results(results)
        
    elif args.operation == 'fix':
        files_modified = normalizer.fix_all_files()
        print_fix_results(files_modified)
        
    elif args.operation == 'verify':
        issues_dict, files_with_issues, total_wrapped, total_blocks = normalizer.verify_all_files()
        print_verify_results(issues_dict, files_with_issues, total_wrapped, total_blocks)
        
    elif args.operation == 'all':
        print("Running complete code block normalization process...\n")
        
        # Step 1: Check
        print("Step 1: Checking for unwrapped code blocks")
        print("-" * 45)
        results = normalizer.check_all_files()
        if results:
            total_unwrapped = sum(len(blocks) for blocks in results.values())
            print(f"Found {total_unwrapped} unwrapped code blocks in {len(results)} files")
        else:
            print("All code blocks are already properly wrapped")
        
        # Step 2: Fix
        print(f"\nStep 2: Fixing code blocks")
        print("-" * 30)
        files_modified = normalizer.fix_all_files()
        if files_modified > 0:
            print(f"Modified {files_modified} files")
        else:
            print("No files needed modification")
        
        # Step 3: Verify
        print(f"\nStep 3: Final verification")
        print("-" * 30)
        issues_dict, files_with_issues, total_wrapped, total_blocks = normalizer.verify_all_files()
        
        if files_with_issues == 0 and total_wrapped == total_blocks:
            print(f"✅ SUCCESS! All {total_blocks} code blocks in {len(list(normalizer.docs_dir.glob('**/*.md')))} files are properly formatted!")
        else:
            print(f"⚠️  Issues remain: {files_with_issues} files have formatting problems")
            for file_path, issues in issues_dict.items():
                print(f"  {file_path}: {len(issues)} issues")


if __name__ == "__main__":
    main()
