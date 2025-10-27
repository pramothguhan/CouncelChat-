"""
Setup script for Legacy's Mental Health CounselChat
Helps initialize the project structure and validate setup
"""

import os
import sys
from pathlib import Path

def create_directory_structure():
    """Create necessary directories for the project."""
    directories = [
        'models',
        'data',
        'outputs',
        'notebooks',
        'src'
    ]
    
    print("📁 Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✓ Created: {directory}/")
    
    print("\n✅ Directory structure created successfully!\n")


def check_env_file():
    """Check if .env file exists and has required keys."""
    print("🔑 Checking environment configuration...")
    
    if not os.path.exists('.env'):
        print("  ⚠️  .env file not found!")
        print("  📝 Creating .env from template...")
        
        if os.path.exists('.env.template'):
            with open('.env.template', 'r') as template:
                with open('.env', 'w') as env_file:
                    env_file.write(template.read())
            print("  ✓ .env file created from template")
            print("\n  ⚠️  IMPORTANT: Edit .env and add your OpenAI API key!")
            return False
        else:
            print("  ❌ .env.template not found!")
            return False
    
    # Check if API key is set
    with open('.env', 'r') as f:
        content = f.read()
        if 'OPENAI_API_KEY' in content:
            print("  ⚠️  API key not configured in .env")
            print("  📝 Please edit .env and add your OpenAI API key")
            return False
        elif 'OPENAI_API_KEY=' in content:
            print("  ✓ .env file exists and appears configured")
            return True
    
    return False


def check_required_files():
    """Check if required model and data files exist."""
    print("\n📦 Checking required files...")
    
    required_files = {
        'models/rag_model_checkpoint.pth': 'Trained RoBERTa model checkpoint',
        'data/counselchat-data.csv': 'CounselChat dataset'
    }
    
    all_present = True
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"  ✓ Found: {file_path} ({file_size:.1f} MB)")
        else:
            print(f"  ❌ Missing: {file_path}")
            print(f"     Description: {description}")
            all_present = False
    
    return all_present


def check_python_version():
    """Check if Python version is compatible."""
    print("\n🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    print("\n📚 Checking dependencies...")
    
    required_packages = [
        'streamlit',
        'torch',
        'transformers',
        'pandas',
        'numpy',
        'sklearn',
        'openai',
        'dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            elif package == 'dotenv':
                __import__('dotenv')
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ❌ {package} (not installed)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n  ⚠️  Missing packages: {', '.join(missing_packages)}")
        print("  📝 Run: pip install -r requirements.txt")
        return False
    
    return True


def create_gitignore():
    """Create .gitignore if it doesn't exist."""
    if not os.path.exists('.gitignore'):
        print("\n📝 Creating .gitignore...")
        gitignore_content = """# Environment Variables
.env
*.env

# Python
__pycache__/
*.py[cod]
*.so

# Models (too large)
*.pth
*.pt

# Data files
*.csv
!data/sample_data.csv

# Outputs
outputs/*.csv

# IDE
.vscode/
.idea/
"""
        with open('.gitignore', 'w') as f:
            f.write(gitignore_content)
        print("  ✓ .gitignore created")


def main():
    """Main setup function."""
    print("="*60)
    print("🧠 Legacy's Mental Health CounselChat - Setup")
    print("="*60)
    print()
    
    # Run all checks
    create_directory_structure()
    python_ok = check_python_version()
    deps_ok = check_dependencies()
    env_ok = check_env_file()
    files_ok = check_required_files()
    create_gitignore()
    
    # Summary
    print("\n" + "="*60)
    print("📊 Setup Summary")
    print("="*60)
    
    checks = [
        ("Python Version", python_ok),
        ("Dependencies", deps_ok),
        ("Environment Config", env_ok),
        ("Required Files", files_ok)
    ]
    
    for check_name, status in checks:
        status_str = "✅ PASS" if status else "❌ FAIL"
        print(f"{check_name:.<40} {status_str}")
    
    all_ok = all(status for _, status in checks)
    
    print("\n" + "="*60)
    if all_ok:
        print("✅ Setup complete! You're ready to run the app.")
        print("\nNext steps:")
        print("  1. Verify your OpenAI API key in .env")
        print("  2. Run: streamlit run app.py")
    else:
        print("⚠️  Setup incomplete. Please address the issues above.")
        print("\nCommon fixes:")
        if not deps_ok:
            print("  • Install dependencies: pip install -r requirements.txt")
        if not env_ok:
            print("  • Configure .env: Add your OpenAI API key")
        if not files_ok:
            print("  • Add required files:")
            print("    - models/rag_model_checkpoint.pth")
            print("    - data/counselchat-data.csv")
    
    print("="*60)


if __name__ == "__main__":
    main()
