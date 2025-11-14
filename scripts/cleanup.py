#!/usr/bin/env python3
"""
Clean up script for Face Recognition System.
Removes temporary files, caches, and old data.
"""

import os
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta


def get_size_mb(path):
    """Get size of file or directory in MB."""
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    elif path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        return total / (1024 * 1024)
    return 0


def clean_pycache(root_dir, dry_run=False):
    """Remove __pycache__ directories."""
    print("\nCleaning __pycache__ directories...")
    removed_count = 0
    total_size = 0
    
    for pycache in Path(root_dir).rglob('__pycache__'):
        if pycache.is_dir():
            size = get_size_mb(pycache)
            total_size += size
            if not dry_run:
                shutil.rmtree(pycache)
            print(f"  {'[DRY RUN] ' if dry_run else ''}Removed: {pycache} ({size:.2f} MB)")
            removed_count += 1
    
    print(f"  Total: {removed_count} directories, {total_size:.2f} MB")
    return removed_count, total_size


def clean_pytest_cache(root_dir, dry_run=False):
    """Remove pytest cache directories."""
    print("\nCleaning pytest cache...")
    removed_count = 0
    total_size = 0
    
    for cache in Path(root_dir).rglob('.pytest_cache'):
        if cache.is_dir():
            size = get_size_mb(cache)
            total_size += size
            if not dry_run:
                shutil.rmtree(cache)
            print(f"  {'[DRY RUN] ' if dry_run else ''}Removed: {cache} ({size:.2f} MB)")
            removed_count += 1
    
    print(f"  Total: {removed_count} directories, {total_size:.2f} MB")
    return removed_count, total_size


def clean_coverage_files(root_dir, dry_run=False):
    """Remove coverage files and directories."""
    print("\nCleaning coverage files...")
    removed_count = 0
    total_size = 0
    
    patterns = ['.coverage', 'coverage.xml', 'htmlcov']
    
    for pattern in patterns:
        for item in Path(root_dir).rglob(pattern):
            size = get_size_mb(item)
            total_size += size
            if not dry_run:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            print(f"  {'[DRY RUN] ' if dry_run else ''}Removed: {item} ({size:.2f} MB)")
            removed_count += 1
    
    print(f"  Total: {removed_count} items, {total_size:.2f} MB")
    return removed_count, total_size


def clean_build_artifacts(root_dir, dry_run=False):
    """Remove build artifacts."""
    print("\nCleaning build artifacts...")
    removed_count = 0
    total_size = 0
    
    patterns = ['build', 'dist', '*.egg-info']
    
    for pattern in patterns:
        for item in Path(root_dir).rglob(pattern):
            if item.is_dir():
                size = get_size_mb(item)
                total_size += size
                if not dry_run:
                    shutil.rmtree(item)
                print(f"  {'[DRY RUN] ' if dry_run else ''}Removed: {item} ({size:.2f} MB)")
                removed_count += 1
    
    print(f"  Total: {removed_count} directories, {total_size:.2f} MB")
    return removed_count, total_size


def clean_logs(root_dir, days=30, dry_run=False):
    """Remove old log files."""
    print(f"\nCleaning log files older than {days} days...")
    removed_count = 0
    total_size = 0
    
    cutoff_date = datetime.now() - timedelta(days=days)
    log_dir = Path(root_dir) / 'logs'
    
    if log_dir.exists():
        for log_file in log_dir.rglob('*.log*'):
            if log_file.is_file():
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                if mtime < cutoff_date:
                    size = get_size_mb(log_file)
                    total_size += size
                    if not dry_run:
                        log_file.unlink()
                    print(f"  {'[DRY RUN] ' if dry_run else ''}Removed: {log_file} ({size:.2f} MB)")
                    removed_count += 1
    
    print(f"  Total: {removed_count} files, {total_size:.2f} MB")
    return removed_count, total_size


def clean_temp_files(root_dir, dry_run=False):
    """Remove temporary files."""
    print("\nCleaning temporary files...")
    removed_count = 0
    total_size = 0
    
    temp_patterns = ['*.pyc', '*.pyo', '*.tmp', '*~', '.DS_Store']
    
    for pattern in temp_patterns:
        for item in Path(root_dir).rglob(pattern):
            if item.is_file():
                size = get_size_mb(item)
                total_size += size
                if not dry_run:
                    item.unlink()
                removed_count += 1
    
    if removed_count > 0:
        print(f"  Total: {removed_count} files, {total_size:.2f} MB")
    else:
        print("  No temporary files found")
    
    return removed_count, total_size


def main():
    """Main cleanup function."""
    parser = argparse.ArgumentParser(description="Clean up Face Recognition System")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be deleted without actually deleting")
    parser.add_argument("--logs-days", type=int, default=30,
                       help="Delete logs older than N days")
    parser.add_argument("--all", action="store_true",
                       help="Clean everything including logs")
    
    args = parser.parse_args()
    
    root_dir = Path(__file__).parent.parent
    
    print("=" * 70)
    print("Face Recognition System - Cleanup")
    print("=" * 70)
    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be deleted ***\n")
    
    total_items = 0
    total_size_mb = 0
    
    # Clean various artifacts
    cleanups = [
        ("Python cache", lambda: clean_pycache(root_dir, args.dry_run)),
        ("Pytest cache", lambda: clean_pytest_cache(root_dir, args.dry_run)),
        ("Coverage files", lambda: clean_coverage_files(root_dir, args.dry_run)),
        ("Build artifacts", lambda: clean_build_artifacts(root_dir, args.dry_run)),
        ("Temporary files", lambda: clean_temp_files(root_dir, args.dry_run)),
    ]
    
    if args.all:
        cleanups.append(("Old logs", lambda: clean_logs(root_dir, args.logs_days, args.dry_run)))
    
    for name, cleanup_func in cleanups:
        count, size = cleanup_func()
        total_items += count
        total_size_mb += size
    
    # Summary
    print("\n" + "=" * 70)
    print("Cleanup Summary")
    print("=" * 70)
    print(f"Total items {'would be ' if args.dry_run else ''}removed: {total_items}")
    print(f"Total space {'would be ' if args.dry_run else ''}freed: {total_size_mb:.2f} MB")
    print("=" * 70)
    
    if args.dry_run:
        print("\nTo actually delete these files, run without --dry-run flag")
    else:
        print("\n✓ Cleanup complete!")


if __name__ == "__main__":
    main()
