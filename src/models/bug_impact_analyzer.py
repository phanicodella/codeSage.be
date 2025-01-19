# Path: codeSage.be/src/models/bug_impact_analyzer.py

import logging
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from pathlib import Path
import ast
from .dci_engine import DCIEngine, CodeInsight

logger = logging.getLogger(__name__)

@dataclass
class BugImpact:
    """Data class for storing bug impact analysis results"""
    severity: str
    confidence: float
    affected_files: List[str]
    affected_functions: List[str]
    potential_fixes: List[str]
    risk_score: float
    priority: str
    estimated_fix_time: str
    test_coverage: Optional[float] = None

class BugImpactAnalyzer:
    """
    Analyzes the potential impact of bugs and suggests fixes based on
    context from the DCI engine.
    """

    def __init__(self, dci_engine: DCIEngine):
        self.dci_engine = dci_engine
        self.current_analysis = {}
        self.bug_patterns = self._load_bug_patterns()

    def analyze_bug(self, 
                   file_path: str, 
                   line_number: int, 
                   bug_description: str = None,
                   stack_trace: str = None) -> BugImpact:
        """
        Analyze a bug's impact and provide detailed insights
        
        Args:
            file_path: Path to the file containing the bug
            line_number: Line number where the bug occurs
            bug_description: Optional description of the bug
            stack_trace: Optional stack trace if available
        """
        try:
            # Get initial impact analysis from DCI engine
            dci_impact = self.dci_engine.get_bug_impact(file_path, line_number)
            
            # Analyze bug context
            context = self._analyze_bug_context(file_path, line_number)
            
            # Determine potential fixes
            fixes = self._suggest_fixes(context, bug_description)
            
            # Calculate comprehensive impact
            severity = self._calculate_severity(dci_impact, context)
            confidence = self._calculate_confidence(context)
            risk_score = self._calculate_risk_score(dci_impact, context)
            
            return BugImpact(
                severity=severity,
                confidence=confidence,
                affected_files=self._get_affected_files(dci_impact),
                affected_functions=self._get_affected_functions(dci_impact),
                potential_fixes=fixes,
                risk_score=risk_score,
                priority=self._determine_priority(severity, risk_score),
                estimated_fix_time=self._estimate_fix_time(context, fixes),
                test_coverage=self._get_test_coverage(file_path)
            )
            
        except Exception as e:
            logger.error(f"Error analyzing bug impact: {str(e)}")
            raise

    def _analyze_bug_context(self, file_path: str, line_number: int) -> Dict:
        """Analyze the context around the bug"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.readlines()
                
            # Get code context (few lines before and after)
            start = max(0, line_number - 5)
            end = min(len(code), line_number + 5)
            context_lines = code[start:end]
            
            # Parse the context
            context_code = ''.join(context_lines)
            tree = ast.parse(context_code)
            
            # Analyze AST for bug patterns
            matched_patterns = []
            for pattern in self.bug_patterns:
                if pattern['detector'](tree):
                    matched_patterns.append(pattern)
            
            return {
                'code_context': context_lines,
                'matched_patterns': matched_patterns,
                'ast_node': self._get_node_at_line(tree, line_number - start)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing bug context: {str(e)}")
            return {}

    def _suggest_fixes(self, context: Dict, bug_description: str = None) -> List[str]:
        """Suggest potential fixes based on context and bug patterns"""
        fixes = []
        
        for pattern in context.get('matched_patterns', []):
            fixes.extend(pattern['fixes'])
            
        if bug_description:
            # Add specific fixes based on bug description
            description_based_fixes = self._analyze_description(bug_description)
            fixes.extend(description_based_fixes)
            
        return list(set(fixes))  # Remove duplicates

    def _calculate_severity(self, dci_impact: Dict, context: Dict) -> str:
        """Calculate bug severity based on impact and context"""
        # Start with DCI impact severity
        base_severity = dci_impact.get('impact', 'low')
        
        # Adjust based on:
        # 1. Number of affected components
        # 2. Data flow impact
        # 3. Matched bug patterns severity
        affected_count = len(dci_impact.get('affected_components', []))
        data_flow_impact = len(dci_impact.get('data_flow_impact', {}).get('affected_variables', []))
        pattern_severity = max(
            (p.get('severity', 0) for p in context.get('matched_patterns', [])),
            default=0
        )
        
        # Calculate weighted severity score
        severity_score = (
            self._severity_to_score(base_severity) * 0.4 +
            min(affected_count / 10, 1.0) * 0.3 +
            min(data_flow_impact / 5, 1.0) * 0.2 +
            pattern_severity * 0.1
        )
        
        return self._score_to_severity(severity_score)

    def _calculate_confidence(self, context: Dict) -> float:
        """Calculate confidence in the analysis"""
        factors = [
            len(context.get('matched_patterns', [])) > 0,  # Pattern matching success
            context.get('ast_node') is not None,  # AST parsing success
            bool(context.get('code_context')),  # Context extraction success
        ]
        return sum(1 for f in factors if f) / len(factors)

    def _calculate_risk_score(self, dci_impact: Dict, context: Dict) -> float:
        """Calculate overall risk score"""
        return min(
            dci_impact.get('risk_score', 0.0) * 0.7 +
            len(context.get('matched_patterns', [])) * 0.15 +
            bool(context.get('ast_node')) * 0.15,
            1.0
        )

    def _determine_priority(self, severity: str, risk_score: float) -> str:
        """Determine fix priority"""
        if severity == 'high' or risk_score > 0.8:
            return 'immediate'
        elif severity == 'medium' or risk_score > 0.5:
            return 'high'
        return 'normal'

    def _estimate_fix_time(self, context: Dict, fixes: List[str]) -> str:
        """Estimate time required to fix the bug"""
        base_time = 30  # Base time in minutes
        
        # Adjust based on:
        # 1. Number of affected components
        # 2. Complexity of fixes
        # 3. Bug pattern complexity
        
        time_multiplier = (
            len(fixes) * 0.5 +
            len(context.get('matched_patterns', [])) * 0.3 +
            bool(context.get('ast_node')) * 0.2
        )
        
        estimated_minutes = base_time * (1 + time_multiplier)
        
        if estimated_minutes < 60:
            return f"{int(estimated_minutes)} minutes"
        else:
            hours = estimated_minutes / 60
            return f"{hours:.1f} hours"

    def _get_test_coverage(self, file_path: str) -> Optional[float]:
        """Get test coverage for the affected file"""
        # This is a placeholder - implement actual test coverage checking
        return None

    @staticmethod
    def _severity_to_score(severity: str) -> float:
        """Convert severity string to numeric score"""
        return {
            'high': 1.0,
            'medium': 0.5,
            'low': 0.2
        }.get(severity, 0.0)

    @staticmethod
    def _score_to_severity(score: float) -> str:
        """Convert numeric score to severity string"""
        if score >= 0.7:
            return 'high'
        elif score >= 0.4:
            return 'medium'
        return 'low'

    @staticmethod
    def _get_node_at_line(tree: ast.AST, line_number: int) -> Optional[ast.AST]:
        """Get AST node at specific line number"""
        for node in ast.walk(tree):
            if hasattr(node, 'lineno') and node.lineno == line_number:
                return node
        return None

    @staticmethod
    def _load_bug_patterns() -> List[Dict]:
        """Load known bug patterns and their fixes"""
        return [
            {
                'name': 'null_pointer',
                'detector': lambda node: any(
                    isinstance(n, ast.Name) and n.id == 'None'
                    for n in ast.walk(node)
                ),
                'severity': 0.8,
                'fixes': ['Add null check before access']
            },
            {
                'name': 'resource_leak',
                'detector': lambda node: any(
                    isinstance(n, ast.With) for n in ast.walk(node)
                ),
                'severity': 0.7,
                'fixes': ['Use context manager (with statement)']
            },
            # Add more patterns as needed
        ]

    def _analyze_description(self, description: str) -> List[str]:
        """Analyze bug description for additional context"""
        fixes = []
        
        # Simple keyword-based analysis
        keywords = {
            'null': ['Add null check', 'Initialize variable'],
            'undefined': ['Check variable existence', 'Add default value'],
            'memory': ['Add resource cleanup', 'Use context manager'],
            'leak': ['Implement proper cleanup', 'Use context manager'],
            'performance': ['Optimize algorithm', 'Add caching'],
            'race condition': ['Add synchronization', 'Use locks'],
            'deadlock': ['Review lock order', 'Use timeout'],
            'timeout': ['Add timeout handling', 'Implement retry logic']
        }
        
        for keyword, suggestions in keywords.items():
            if keyword in description.lower():
                fixes.extend(suggestions)
        
        return fixes

    def _get_affected_files(self, dci_impact: Dict) -> List[str]:
        """Get list of affected files"""
        return [
            comp.split('::')[0] 
            for comp in dci_impact.get('affected_components', [])
        ]

    def _get_affected_functions(self, dci_impact: Dict) -> List[str]:
        """Get list of affected functions"""
        return [
            comp.split('::')[1] 
            for comp in dci_impact.get('affected_components', [])
            if '::' in comp
        ]