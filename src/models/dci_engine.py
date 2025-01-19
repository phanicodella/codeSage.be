# Path: codeSage.be/src/models/dci_engine.py

import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
import ast
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CodeInsight:
    """Data class for storing code insights"""
    function_name: str
    dependencies: List[str]
    complexity: int
    impact_score: float
    called_by: List[str]
    calls_to: List[str]
    variables_used: List[str]
    data_flow: Dict[str, List[str]]

class DCIEngine:
    """
    Deep Contextual Insight Engine for analyzing code relationships,
    dependencies, and potential impact of changes.
    """

    def __init__(self):
        self.dependency_graph = nx.DiGraph()
        self.call_graph = nx.DiGraph()
        self.data_flow_graph = nx.DiGraph()

    def analyze_codebase(self, root_path: Path) -> Dict[str, CodeInsight]:
        """Analyze entire codebase for deep insights"""
        insights = {}
        
        try:
            for file_path in root_path.rglob('*.py'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                file_insights = self._analyze_file(code, str(file_path))
                insights.update(file_insights)

            # Build relationships between components
            self._build_dependency_graph(insights)
            self._analyze_data_flow(insights)
            self._calculate_impact_scores(insights)

            return insights
        except Exception as e:
            logger.error(f"Error analyzing codebase: {str(e)}")
            raise

    def get_bug_impact(self, file_path: str, line_number: int) -> Dict:
        """Analyze potential impact of a bug at specific location"""
        try:
            # Get affected component
            component = self._get_component_at_location(file_path, line_number)
            if not component:
                return {"impact": "low", "affected_components": []}

            # Find potentially affected components
            affected = self._get_affected_components(component)
            severity = self._calculate_severity(affected)

            return {
                "impact": severity,
                "affected_components": list(affected),
                "risk_score": self._calculate_risk_score(affected),
                "suggested_fix_priority": self._determine_priority(severity),
                "data_flow_impact": self._analyze_data_flow_impact(component)
            }
        except Exception as e:
            logger.error(f"Error analyzing bug impact: {str(e)}")
            return {"error": str(e)}

    def _analyze_file(self, code: str, file_path: str) -> Dict[str, CodeInsight]:
        """Analyze a single file for insights"""
        insights = {}
        try:
            tree = ast.parse(code)
            analyzer = CodeAnalyzer()
            analyzer.visit(tree)

            for func_name, details in analyzer.functions.items():
                insights[f"{file_path}::{func_name}"] = CodeInsight(
                    function_name=func_name,
                    dependencies=details['dependencies'],
                    complexity=details['complexity'],
                    impact_score=0.0,  # Will be calculated later
                    called_by=[],
                    calls_to=details['calls'],
                    variables_used=details['variables'],
                    data_flow=details['data_flow']
                )

            return insights
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {str(e)}")
            return {}

    def _build_dependency_graph(self, insights: Dict[str, CodeInsight]):
        """Build dependency graph from insights"""
        for component, insight in insights.items():
            self.dependency_graph.add_node(component)
            for dep in insight.dependencies:
                self.dependency_graph.add_edge(component, dep)

    def _analyze_data_flow(self, insights: Dict[str, CodeInsight]):
        """Analyze data flow between components"""
        for component, insight in insights.items():
            for var, flows in insight.data_flow.items():
                for flow in flows:
                    self.data_flow_graph.add_edge(component, flow, variable=var)

    def _calculate_impact_scores(self, insights: Dict[str, CodeInsight]):
        """Calculate impact scores for each component"""
        for component in insights:
            # Calculate based on:
            # 1. Number of dependent components
            # 2. Complexity of the component
            # 3. Data flow connections
            dependencies = list(nx.descendants(self.dependency_graph, component))
            data_flows = list(nx.descendants(self.data_flow_graph, component))
            
            impact_score = (
                len(dependencies) * 0.4 +
                insights[component].complexity * 0.3 +
                len(data_flows) * 0.3
            )
            
            insights[component].impact_score = min(impact_score, 1.0)

    def _get_component_at_location(self, file_path: str, line_number: int) -> Optional[str]:
        """Get component identifier at specific file location"""
        # Implementation depends on how you store file locations
        return None  # Placeholder

    def _get_affected_components(self, component: str) -> Set[str]:
        """Get components that might be affected by changes to given component"""
        affected = set()
        
        # Check dependency graph
        affected.update(nx.descendants(self.dependency_graph, component))
        
        # Check data flow graph
        affected.update(nx.descendants(self.data_flow_graph, component))
        
        return affected

    def _calculate_severity(self, affected_components: Set[str]) -> str:
        """Calculate severity based on affected components"""
        if len(affected_components) > 10:
            return "high"
        elif len(affected_components) > 5:
            return "medium"
        return "low"

    def _calculate_risk_score(self, affected_components: Set[str]) -> float:
        """Calculate risk score based on affected components"""
        if not affected_components:
            return 0.0
            
        total_score = 0
        for component in affected_components:
            if component in self.dependency_graph:
                # Consider:
                # 1. Number of incoming dependencies
                # 2. Number of outgoing dependencies
                # 3. Betweenness centrality
                in_deg = self.dependency_graph.in_degree(component)
                out_deg = self.dependency_graph.out_degree(component)
                centrality = nx.betweenness_centrality(self.dependency_graph)[component]
                
                component_score = (in_deg * 0.4 + out_deg * 0.3 + centrality * 0.3)
                total_score += component_score

        return min(total_score / len(affected_components), 1.0)

    def _determine_priority(self, severity: str) -> str:
        """Determine fix priority based on severity"""
        priority_mapping = {
            "high": "immediate",
            "medium": "high",
            "low": "normal"
        }
        return priority_mapping.get(severity, "normal")

    def _analyze_data_flow_impact(self, component: str) -> Dict:
        """Analyze impact on data flow"""
        if component not in self.data_flow_graph:
            return {"affected_variables": [], "flow_paths": []}

        affected_vars = set()
        flow_paths = []

        # Find all paths in data flow graph
        for successor in nx.descendants(self.data_flow_graph, component):
            paths = list(nx.all_simple_paths(self.data_flow_graph, component, successor))
            for path in paths:
                variables = [
                    self.data_flow_graph[path[i]][path[i+1]]['variable']
                    for i in range(len(path)-1)
                ]
                affected_vars.update(variables)
                flow_paths.append({
                    "path": path,
                    "variables": variables
                })

        return {
            "affected_variables": list(affected_vars),
            "flow_paths": flow_paths
        }

class CodeAnalyzer(ast.NodeVisitor):
    """AST visitor for analyzing Python code"""
    
    def __init__(self):
        self.functions = {}
        self.current_function = None
        self.complexity = 0

    def visit_FunctionDef(self, node):
        self.current_function = node.name
        self.complexity = 1
        self.functions[node.name] = {
            'dependencies': [],
            'complexity': 0,
            'calls': [],
            'variables': [],
            'data_flow': {}
        }
        self.generic_visit(node)
        self.functions[node.name]['complexity'] = self.complexity
        self.current_function = None

    def visit_Call(self, node):
        if self.current_function:
            if isinstance(node.func, ast.Name):
                self.functions[self.current_function]['calls'].append(node.func.id)
        self.generic_visit(node)

    def visit_Name(self, node):
        if self.current_function and isinstance(node.ctx, ast.Load):
            self.functions[self.current_function]['variables'].append(node.id)
        self.generic_visit(node)

    def visit_If(self, node):
        if self.current_function:
            self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node):
        if self.current_function:
            self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node):
        if self.current_function:
            self.complexity += 1
        self.generic_visit(node)