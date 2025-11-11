#!/usr/bin/env python3
"""
Changelog Entry Generator for Face Recognition System.

This script helps developers create consistent changelog entries by parsing
recent git commits and generating formatted markdown.

It follows the "Keep a Changelog" format and categorizes entries based on
conventional commit prefixes (e.g., feat, fix, docs, chore).

Dependencies:
- GitPython
- click

Usage:
  python scripts/gen_changelog.py --rev-range <commit1>..<commit2>
  python scripts/gen_changelog.py --from-tag v1.0.0
"""

import re
import click
from git import Repo, GitCommandError
from datetime import datetime

# Conventional Commit Types to Changelog Sections
COMMIT_TYPE_MAP = {
    "feat": "Added",
    "fix": "Fixed",
    "perf": "Changed",
    "refactor": "Changed",
    "docs": "Documentation",
    "style": "Style",
    "test": "Testing",
    "build": "Build",
    "ci": "CI/CD",
    "chore": "Chore",
}

def get_commits(repo_path, rev_range=None, from_tag=None):
    """Get a list of commits from the repository."""
    try:
        repo = Repo(repo_path, search_parent_directories=True)
        
        if from_tag:
            # Get commits since the specified tag
            if from_tag not in repo.tags:
                raise ValueError(f"Tag '{from_tag}' not found.")
            rev_range = f"{from_tag}..HEAD"
        
        if not rev_range:
            # Default to commits since the last tag
            tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime)
            if tags:
                rev_range = f"{tags[-1].name}..HEAD"
            else:
                # If no tags, get all commits
                rev_range = None

        commits = list(repo.iter_commits(rev_range))
        if not commits:
            click.echo("No new commits found in the specified range.", err=True)
            return []
            
        return commits
        
    except GitCommandError as e:
        click.echo(f"Git error: {e}", err=True)
        return []
    except Exception as e:
        click.echo(f"An error occurred: {e}", err=True)
        return []

def parse_commit_message(message):
    """Parse a commit message to extract type, scope, and subject."""
    match = re.match(r"(\w+)(?:\(([\w\-/]+)\))?:\s(.+)", message)
    if match:
        return match.groups()
    return None, None, message

def generate_changelog_entries(commits):
    """Generate changelog entries from a list of commits."""
    entries = {section: [] for section in COMMIT_TYPE_MAP.values()}
    
    for commit in commits:
        commit_type, scope, subject = parse_commit_message(commit.summary)
        
        section = COMMIT_TYPE_MAP.get(commit_type, "Changed")
        
        entry = f"- {subject}"
        if scope:
            entry += f" (**{scope}**)"
        
        entries[section].append(entry)
        
    return entries

def format_changelog(entries, version="Unreleased"):
    """Format the changelog entries into markdown."""
    output = [f"## [{version}] - {datetime.now().strftime('%Y-%m-%d')}\n"]
    
    for section, items in entries.items():
        if items:
            output.append(f"### {section}")
            output.extend(sorted(list(set(items))))
            output.append("")
            
    return "\n".join(output)

@click.command()
@click.option('--repo-path', default='.', help='Path to the git repository.')
@click.option('--rev-range', help='Git revision range (e.g., "main..develop", "HEAD~5..HEAD").')
@click.option('--from-tag', help='Generate changelog since this git tag.')
@click.option('--output', '-o', type=click.File('w'), default='-', help='Output file (or - for stdout).')
def main(repo_path, rev_range, from_tag, output):
    """Generates a changelog from git commits."""
    
    commits = get_commits(repo_path, rev_range, from_tag)
    
    if not commits:
        return
        
    entries = generate_changelog_entries(commits)
    changelog_md = format_changelog(entries)
    
    click.echo(changelog_md, file=output)
    if output.name != '<stdout>':
        click.echo(f"Changelog written to {output.name}")

if __name__ == "__main__":
    main()
