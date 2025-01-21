# Path: codeSage.be/src/services/file_crawler.py

import os
from pathlib import Path
from typing import List, Dict, Set, Generator
import logging
import fnmatch
from concurrent.futures import ThreadPoolExecutor, as_completed
import chardet
from dataclasses import dataclass
from datetime import datetime
import git
from git.exc import InvalidGitRepositoryError

logger = logging.getLogger(__name__)

@dataclass
class FileMetadata:
    """Data class to store file metadata information"""
    path: str
    size: int
    last_modified: datetime
    extension: str
    is_binary: bool
    git_history: Dict = None
    encoding: str = 'utf-8'

class FileCrawler:
    """
    Service for crawling and analyzing codebases, providing detailed file information
    and content analysis capabilities.
    """

    def __init__(self):
        """
        Initialize the FileCrawler service.
        No artificial size limits - system resources are the only constraint.
        """
        self.binary_extensions = {
            '.pyc', '.pyo', '.so', '.dll', '.dylib', '.jar', '.war',
            '.ear', '.class', '.exe', '.pkl', '.bin', '.dat', '.db',
            '.sqlite', '.sqlite3', '.gif', '.jpg', '.jpeg', '.png',
            '.bmp', '.ico', '.pdf', '.zip', '.tar', '.gz', '.rar'
        }
        self.ignore_patterns = [
            '**/.git/**', '**/node_modules/**', '**/__pycache__/**',
            '**/venv/**', '**/env/**', '**/build/**', '**/dist/**',
            '**/.idea/**', '**/.vscode/**', '**/coverage/**'
        ]

    def should_ignore(self, path: str) -> bool:
        """
        Check if a path should be ignored based on patterns.

        Args:
            path: File or directory path to check

        Returns:
            Boolean indicating whether the path should be ignored
        """
        return any(fnmatch.fnmatch(path, pattern) for pattern in self.ignore_patterns)

    def is_binary_file(self, file_path: str) -> bool:
        """
        Determine if a file is binary based on extension or content sampling.

        Args:
            file_path: Path to the file

        Returns:
            Boolean indicating whether the file is binary
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext in self.binary_extensions:
            return True
        
        try:
            with open(file_path, 'rb') as f:
                sample = f.read(1024)
                return b'\x00' in sample
        except Exception as e:
            logger.error(f"Error checking if file is binary: {str(e)}")
            return True

    def get_file_metadata(self, file_path: str, include_git: bool = True) -> FileMetadata:
        """
        Get detailed metadata for a file.

        Args:
            file_path: Path to the file
            include_git: Whether to include git history information

        Returns:
            FileMetadata object containing file information
        """
        path_obj = Path(file_path)
        stats = path_obj.stat()
        
        # Detect file encoding
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(4096)
                result = chardet.detect(raw_data)
                encoding = result['encoding'] if result['encoding'] else 'utf-8'
        except Exception as e:
            logger.warning(f"Error detecting encoding for {file_path}: {str(e)}")
            encoding = 'utf-8'

        metadata = FileMetadata(
            path=str(path_obj),
            size=stats.st_size,
            last_modified=datetime.fromtimestamp(stats.st_mtime),
            extension=path_obj.suffix.lower(),
            is_binary=self.is_binary_file(file_path),
            encoding=encoding
        )

        if include_git:
            try:
                repo = git.Repo(path_obj.parent, search_parent_directories=True)
                commits = list(repo.iter_commits(paths=file_path, max_count=10))
                metadata.git_history = {
                    'last_commit_hash': str(commits[0]) if commits else None,
                    'commit_count': len(commits),
                    'authors': list(set(c.author.name for c in commits))
                }
            except (InvalidGitRepositoryError, Exception) as e:
                logger.debug(f"No git history for {file_path}: {str(e)}")
                metadata.git_history = None

        return metadata

    def read_file_content(self, file_path: str) -> str:
        """
        Read and return file content with appropriate encoding detection.

        Args:
            file_path: Path to the file

        Returns:
            String containing file content
        """
        metadata = self.get_file_metadata(file_path, include_git=False)
        if metadata.is_binary:
            raise ValueError(f"Cannot read binary file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding=metadata.encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            # Fallback to binary read and decode
            with open(file_path, 'rb') as f:
                content = f.read()
                return content.decode(errors='replace')

    def crawl_directory(self, 
                       root_path: str, 
                       file_extensions: Set[str] = None,
                       max_threads: int = 4) -> Generator[FileMetadata, None, None]:
        """
        Crawl a directory and yield FileMetadata for each file found.

        Args:
            root_path: Root directory to start crawling from
            file_extensions: Set of file extensions to include (None for all)
            max_threads: Maximum number of threads for parallel processing

        Yields:
            FileMetadata objects for each file processed
        """
        root_path = os.path.abspath(root_path)
        if not os.path.exists(root_path):
            raise ValueError(f"Directory does not exist: {root_path}")

        def process_file(file_path: str) -> FileMetadata:
            try:
                return self.get_file_metadata(file_path)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                return None

        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            future_to_path = {}
            
            for root, _, files in os.walk(root_path):
                if self.should_ignore(root):
                    continue
                    
                for file in files:
                    file_path = os.path.join(root, file)
                    if self.should_ignore(file_path):
                        continue
                        
                    if file_extensions and os.path.splitext(file)[1].lower() not in file_extensions:
                        continue
                        
                    future_to_path[executor.submit(process_file, file_path)] = file_path

            for future in as_completed(future_to_path):
                file_path = future_to_path[future]
                try:
                    metadata = future.result()
                    if metadata:
                        yield metadata
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")

    def get_project_summary(self, root_path: str) -> Dict:
        """
        Generate a summary of the project directory.

        Args:
            root_path: Root directory of the project

        Returns:
            Dictionary containing project summary information
        """
        total_files = 0
        total_size = 0
        extensions = {}
        languages = set()
        
        for metadata in self.crawl_directory(root_path):
            total_files += 1
            total_size += metadata.size
            ext = metadata.extension
            extensions[ext] = extensions.get(ext, 0) + 1
            
            # Language detection based on extension
            if ext in {'.py', '.pyw'}:
                languages.add('Python')
            elif ext in {'.js', '.jsx', '.ts', '.tsx'}:
                languages.add('JavaScript/TypeScript')
            elif ext in {'.java'}:
                languages.add('Java')
            elif ext in {'.cpp', '.hpp', '.c', '.h'}:
                languages.add('C/C++')

        return {
            'total_files': total_files,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'file_extensions': extensions,
            'detected_languages': list(languages),
            'crawl_timestamp': datetime.now().isoformat()
        }