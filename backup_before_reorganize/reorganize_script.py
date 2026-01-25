#!/usr/bin/env python3
"""
Project Reorganization Script
Reorganizes the Solana copy trading bot into a clean folder structure
"""

import os
import shutil
from pathlib import Path

# Define the new structure
STRUCTURE = {
    'core': [
        'master_v2.py',
        'database_v2.py',
        'strategist_v2.py',
        'improved_discovery_v8.py',
        'discovery_config.py',
        'discovery_integration.py',
        'profiler.py',
        'effective_paper_trader.py',
        'paper_engine_replacement.py',
    ],
    'infrastructure': [
        'multi_webhook_manager.py',
        'helius_webhook_manager.py',
    ],
    'cli': [
        'run_discovery.py',
        'system_analysis.py',
        'discovery_dashboard.py',
        'analyze_positions.py',
    ],
    'diagnostics': [
        'diagnose_discovery.py',
        'diagnose_issues.py',
        'diagnose_prefilter.py',
        'debug_diagnostic.py',
        'debug_profiler.py',
        'discover_endpoints.py',
        'test_discovery.py',
        'test_api.py',
    ],
    'setup': [
        'migrate_paper_trader.py',
        'seed_wallets_script.py',
        'register_webhook.py',
        'fix_webhooks.py',
    ],
}

# Files to keep in root
ROOT_FILES = [
    '.env.example',
    '.gitignore',
    'requirements.txt',
    'README.md',
    'LICENSE',
]

# Files/folders to ignore
IGNORE = [
    '.git',
    '.env',
    '__pycache__',
    '*.pyc',
    'swing_traders.db',
    'paper_trades_v3.db',
    'discovery_debug.json',
    'venv',
    'env',
    '.vscode',
    '.idea',
]


def should_ignore(path):
    """Check if path should be ignored"""
    name = os.path.basename(path)
    for pattern in IGNORE:
        if pattern.startswith('*'):
            if name.endswith(pattern[1:]):
                return True
        elif pattern == name:
            return True
    return False


def create_backup():
    """Create a backup of the current structure"""
    print("\n📦 Creating backup...")
    backup_dir = Path('backup_before_reorganize')
    
    if backup_dir.exists():
        print(f"   Backup already exists at {backup_dir}")
        response = input("   Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("   Skipping backup")
            return None
        shutil.rmtree(backup_dir)
    
    backup_dir.mkdir()
    
    # Copy all .py files to backup
    for file in Path('.').glob('*.py'):
        if not should_ignore(file):
            shutil.copy2(file, backup_dir / file.name)
            print(f"   ✅ Backed up: {file.name}")
    
    print(f"\n   Backup created at: {backup_dir}/")
    return backup_dir


def create_directories():
    """Create the new directory structure"""
    print("\n📁 Creating directory structure...")
    
    for folder in STRUCTURE.keys():
        folder_path = Path(folder)
        if not folder_path.exists():
            folder_path.mkdir()
            print(f"   ✅ Created: {folder}/")
        else:
            print(f"   ℹ️  Already exists: {folder}/")
        
        # Create __init__.py for Python packages
        init_file = folder_path / '__init__.py'
        if not init_file.exists():
            init_file.touch()
            print(f"   ✅ Created: {folder}/__init__.py")


def move_files():
    """Move files to their new locations"""
    print("\n🚚 Moving files...")
    
    moved = []
    missing = []
    
    for folder, files in STRUCTURE.items():
        for file in files:
            source = Path(file)
            destination = Path(folder) / file
            
            if source.exists():
                if destination.exists():
                    print(f"   ⚠️  Already exists: {destination}")
                else:
                    shutil.move(str(source), str(destination))
                    print(f"   ✅ Moved: {file} → {folder}/")
                    moved.append((file, folder))
            else:
                print(f"   ⚠️  Not found: {file}")
                missing.append(file)
    
    return moved, missing


def update_imports_in_file(file_path, moved_files):
    """Update import statements in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes = []
        
        # Build import mapping
        import_map = {}
        for file, folder in moved_files:
            module_name = file.replace('.py', '')
            import_map[module_name] = f"{folder}.{module_name}"
        
        # Update imports
        for old_module, new_module in import_map.items():
            # Handle: from module import ...
            old_import = f"from {old_module} import"
            new_import = f"from {new_module} import"
            if old_import in content:
                content = content.replace(old_import, new_import)
                changes.append(f"{old_import} → {new_import}")
            
            # Handle: import module
            old_import = f"import {old_module}"
            new_import = f"import {new_module}"
            if old_import in content:
                content = content.replace(old_import, new_import)
                changes.append(f"{old_import} → {new_import}")
        
        # Save if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return changes
        return []
        
    except Exception as e:
        print(f"   ⚠️  Error updating {file_path}: {e}")
        return []


def update_all_imports(moved_files):
    """Update import statements in all Python files"""
    print("\n🔧 Updating import statements...")
    
    total_changes = 0
    
    # Update files in new structure
    for folder in STRUCTURE.keys():
        folder_path = Path(folder)
        for py_file in folder_path.glob('*.py'):
            if py_file.name == '__init__.py':
                continue
            
            changes = update_imports_in_file(py_file, moved_files)
            if changes:
                print(f"\n   📝 {py_file}:")
                for change in changes:
                    print(f"      {change}")
                total_changes += len(changes)
    
    print(f"\n   ✅ Updated {total_changes} import statement(s)")


def create_main_launcher():
    """Create a main.py launcher in root"""
    print("\n🚀 Creating main launcher...")
    
    launcher_content = '''#!/usr/bin/env python3
"""
Main launcher for Solana Copy Trading Bot
Run this file to start the system
"""

import sys
from pathlib import Path

# Add core directory to path
sys.path.insert(0, str(Path(__file__).parent / 'core'))
sys.path.insert(0, str(Path(__file__).parent / 'infrastructure'))

# Import and run master
from master_v2 import main

if __name__ == "__main__":
    main()
'''
    
    launcher_path = Path('main.py')
    with open(launcher_path, 'w') as f:
        f.write(launcher_content)
    
    # Make executable on Unix systems
    try:
        os.chmod(launcher_path, 0o755)
    except:
        pass
    
    print(f"   ✅ Created: main.py")


def create_updated_readme_section():
    """Create content for updating README"""
    readme_section = '''

## 📁 Updated Project Structure

```
solana-copy-trading-bot/
├── core/                          # Core system components
│   ├── master_v2.py              # Main orchestrator
│   ├── database_v2.py            # Database layer
│   ├── strategist_v2.py          # Strategy engine
│   ├── improved_discovery_v8.py  # Discovery system
│   ├── discovery_config.py       # Discovery config
│   ├── discovery_integration.py  # Discovery integration
│   ├── profiler.py               # Wallet profiling
│   ├── effective_paper_trader.py # Paper trading engine
│   └── paper_engine_replacement.py
│
├── infrastructure/               # Infrastructure & scaling
│   ├── multi_webhook_manager.py # Multi-webhook support
│   └── helius_webhook_manager.py
│
├── cli/                         # Command-line tools
│   ├── run_discovery.py        # Discovery CLI
│   ├── system_analysis.py      # System validation
│   ├── discovery_dashboard.py  # Discovery monitoring
│   └── analyze_positions.py    # Position analysis
│
├── diagnostics/                # Debugging tools
│   ├── diagnose_discovery.py
│   ├── diagnose_issues.py
│   ├── diagnose_prefilter.py
│   ├── debug_diagnostic.py
│   ├── debug_profiler.py
│   ├── discover_endpoints.py
│   ├── test_discovery.py
│   └── test_api.py
│
├── setup/                      # One-time setup scripts
│   ├── migrate_paper_trader.py
│   ├── seed_wallets_script.py
│   ├── register_webhook.py
│   └── fix_webhooks.py
│
├── main.py                     # Main launcher (run this!)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Running the Bot

After reorganization, use the main launcher:

```bash
# Start the bot
python main.py

# Or run CLI tools
python cli/run_discovery.py
python cli/system_analysis.py
python cli/discovery_dashboard.py summary 7
python cli/analyze_positions.py

# Or run diagnostics
python diagnostics/diagnose_discovery.py
python diagnostics/test_api.py
```
'''
    
    return readme_section


def show_summary(moved, missing, backup_dir):
    """Show summary of changes"""
    print("\n" + "="*70)
    print("📊 REORGANIZATION SUMMARY")
    print("="*70)
    
    print(f"\n✅ Successfully moved {len(moved)} file(s)")
    print(f"📁 Created {len(STRUCTURE)} directories")
    
    if missing:
        print(f"\n⚠️  {len(missing)} file(s) not found:")
        for file in missing:
            print(f"   - {file}")
    
    if backup_dir:
        print(f"\n💾 Backup created at: {backup_dir}/")
    
    print("\n📝 Next steps:")
    print("   1. Test the new structure: python main.py")
    print("   2. Update your .gitignore if needed")
    print("   3. Test CLI tools: python cli/run_discovery.py")
    print("   4. Update README.md with new structure (content provided below)")
    print("   5. If everything works, delete backup_before_reorganize/")
    
    print("\n" + "="*70)
    print("README UPDATE CONTENT")
    print("="*70)
    print(create_updated_readme_section())
    print("="*70)


def main():
    """Main reorganization process"""
    print("\n" + "="*70)
    print("🔧 SOLANA COPY TRADING BOT - PROJECT REORGANIZATION")
    print("="*70)
    
    print("\nThis script will:")
    print("  1. Create a backup of all .py files")
    print("  2. Create new directory structure")
    print("  3. Move files to appropriate folders")
    print("  4. Update import statements")
    print("  5. Create a main.py launcher")
    
    response = input("\nContinue? (y/n): ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return
    
    # Step 1: Backup
    backup_dir = create_backup()
    
    # Step 2: Create directories
    create_directories()
    
    # Step 3: Move files
    moved, missing = move_files()
    
    # Step 4: Update imports
    update_all_imports(moved)
    
    # Step 5: Create launcher
    create_main_launcher()
    
    # Show summary
    show_summary(moved, missing, backup_dir)
    
    print("\n✅ Reorganization complete!")


if __name__ == "__main__":
    main()
