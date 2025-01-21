import os
import fnmatch
from pathlib import Path


def should_exclude_dir(dir_name):
    """Check if directory should be excluded."""
    exclude_patterns = {
        'node_modules',
        'venv',
        '.git',
        'build',
        'dist',
        'bin',
        'obj',
        '__pycache__',
        '.vs',
        '.idea',
        'packages',
        'vendor',
        'bower_components',
        'jspm_packages',
        'lib',
        'out',
        'target',
        'Debug',
        'Release'
    }
    return any(dir_name.lower().startswith(pattern) for pattern in exclude_patterns)


def is_source_file(filename):
    """Check if file is a source code file and not unnecessary manifests or configs."""
    source_patterns = [
        '*.js', '*.jsx', '*.ts', '*.tsx',  # JavaScript/TypeScript
        '*.py',                            # Python
        '*.cs',                            # C#
        '*.css', '*.scss', '*.sass',       # Stylesheets
        '*.html', '*.htm',                 # HTML
        '*.java',                          # Java
        '*.cpp', '*.hpp', '*.c', '*.h',    # C/C++
        '*.go',                            # Go
        '*.rb',                            # Ruby
        '*.php',                           # PHP
        '*.swift',                         # Swift
        '*.rs',                            # Rust
        '*.vue', '*.svelte',               # Web frameworks
        '*.xml', '*.json',                 # Data formats
        '*.yaml', '*.yml'                  # Configuration files
    ]
    exclude_files = [
        '*-lock.json',                     # Exclude lock files
        '*-weights_manifest.json',         # Manifest files
        '*.eslint*',                       # ESLint configuration
        '*.prettier*',                     # Prettier configuration
        '*.log', 
        '*.lock',                           # Log files
        'README.md',                       # Documentation
        '*.md'                             # Other markdown files
    ]
    include_files = [
        'package.json'                     # Explicitly include package.json
    ]


    # Match only source files and exclude unnecessary files
    if filename in include_files:
        return True
    return (
        any(fnmatch.fnmatch(filename.lower(), pattern) for pattern in source_patterns) and
        not any(fnmatch.fnmatch(filename.lower(), exclude) for exclude in exclude_files)
    )


def capture_source_code(root_dir='.', output_file='CompleteSourceCode.txt'):
    """
    Capture all source code files from the project directory.


    Args:
        root_dir (str): Root directory to start scanning from
        output_file (str): Output file path
    """
    root_path = Path(root_dir).absolute()


    with open(output_file, 'w', encoding='utf-8') as f:
        for root, dirs, files in os.walk(root_dir):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if not should_exclude_dir(d)]


            for file in files:
                if is_source_file(file):
                    file_path = Path(root) / file
                    try:
                        # Use os.path.relpath for more reliable path calculation
                        relative_path = os.path.relpath(file_path, root_path)


                        with open(file_path, 'r', encoding='utf-8') as source_file:
                            content = source_file.read()


                            # Write file path and content with separator
                            f.write(f"filepath:///{relative_path} /// /// ///\n")
                            f.write("file code{\n")
                            f.write(content)
                            f.write("\n}\n\n")


                    except Exception as e:
                        print(f"Error processing file {file_path}: {str(e)}")




if __name__ == "__main__":
    try:
        current_dir = os.getcwd()
        capture_source_code(current_dir)
        print("Source code capture completed successfully!")
        print("Output saved to: CompleteSourceCode.txt")
    except Exception as e:
        print(f"An error occurred: {str(e)}")