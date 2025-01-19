# File path: E:\codeSage\codeSage.be\src\models\analyzer.py

import ast
from typing import Dict, List, Any
from dataclasses import dataclass
import logging

@dataclass
class AnalysisResult:
    """Data class to store analysis results"""
    file_path: str
    complexity: int
    imports: List[str]
    classes: List[str]
    functions: List[str]
    potential_issues: List[Dict[str, Any]]

class CodeAnalyzer:
    """Code analyzer for Python files"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_file(self, file_path: str) -> AnalysisResult:
        """
        Analyze a Python file and return its metrics
        Args:
            file_path: Path to the Python file
        Returns:
            AnalysisResult object containing analysis metrics
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            tree = ast.parse(content)
            
            return AnalysisResult(
                file_path=file_path,
                complexity=self._calculate_complexity(tree),
                imports=self._get_imports(tree),
                classes=self._get_classes(tree),
                functions=self._get_functions(tree),
                potential_issues=self._find_potential_issues(tree)
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {str(e)}")
            raise
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
                
        return complexity
    
    def _get_imports(self, tree: ast.AST) -> List[str]:
        """Extract all imports from the AST"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for name in node.names:
                    imports.append(f"{module}.{name.name}")
                    
        return imports
    
    def _get_classes(self, tree: ast.AST) -> List[str]:
        """Extract all class names from the AST"""
        return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    
    def _get_functions(self, tree: ast.AST) -> List[str]:
        """Extract all function names from the AST"""
        return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    
    def _find_potential_issues(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Identify potential code issues"""
        issues = []
        
        for node in ast.walk(tree):
            # Check for overly complex functions
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_complexity(node)
                if complexity > 10:
                    issues.append({
                        'type': 'high_complexity',
                        'element': node.name,
                        'complexity': complexity,
                        'line': node.lineno
                    })
            
            # Check for bare excepts
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append({
                    'type': 'bare_except',
                    'line': node.lineno
                })
            
            # Check for potentially dangerous globals
            if isinstance(node, ast.Global):
                issues.append({
                    'type': 'global_usage',
                    'names': node.names,
                    'line': node.lineno
                })
        
        return issues