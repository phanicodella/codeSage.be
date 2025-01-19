# Path: codeSage.be/src/services/visualization_service.py

import networkx as nx
from typing import Dict, List, Any, Optional
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import io
import base64
import plotly.graph_objects as go
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for visualization customization"""
    theme: str = "dark"
    node_size: int = 20
    edge_width: float = 1.0
    font_size: int = 12
    layout: str = "force"  # force, circular, hierarchical
    show_labels: bool = True
    show_weights: bool = False

class VisualizationService:
    """
    Service for creating interactive visualizations of code relationships,
    dependencies, and data flow.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self._setup_themes()

    def _setup_themes(self):
        """Setup color themes for visualizations"""
        self.themes = {
            "dark": {
                "background": "#1a1a1a",
                "nodes": "#4a90e2",
                "edges": "#666666",
                "text": "#ffffff",
                "highlight": "#ff6b6b"
            },
            "light": {
                "background": "#ffffff",
                "nodes": "#2c5282",
                "edges": "#cbd5e0",
                "text": "#1a202c",
                "highlight": "#e53e3e"
            }
        }

    def create_dependency_graph(self, dependencies: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Create interactive dependency graph visualization
        
        Args:
            dependencies: Dictionary mapping components to their dependencies
            
        Returns:
            Dictionary containing the Plotly figure data
        """
        G = nx.DiGraph()
        
        # Add nodes and edges
        for source, targets in dependencies.items():
            G.add_node(source)
            for target in targets:
                G.add_edge(source, target)

        # Calculate layout
        if self.config.layout == "force":
            pos = nx.spring_layout(G)
        elif self.config.layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.kamada_kawai_layout(G)

        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text' if self.config.show_labels else 'markers',
            hoverinfo='text',
            text=node_text,
            textposition="bottom center",
            marker={
                'size': self.config.node_size,
                'color': self.themes[self.config.theme]["nodes"],
                'line_width': 2
            }
        )

        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=self.config.edge_width, color=self.themes[self.config.theme]["edges"]),
            hoverinfo='none',
            mode='lines'
        )

        # Create figure
        figure = {
            'data': [edge_trace, node_trace],
            'layout': go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Dependency Graph",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    font=dict(size=16)
                ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor=self.themes[self.config.theme]["background"],
                paper_bgcolor=self.themes[self.config.theme]["background"],
                font=dict(color=self.themes[self.config.theme]["text"])
            )
        }

        return figure

    def create_data_flow_diagram(self, data_flows: Dict[str, Dict[str, List[str]]]) -> Dict[str, Any]:
        """Create interactive data flow visualization"""
        G = nx.DiGraph()
        
        # Create nodes and edges for data flow
        for component, flows in data_flows.items():
            G.add_node(component, type="component")
            for var, targets in flows.items():
                var_node = f"{component}::{var}"
                G.add_node(var_node, type="variable")
                G.add_edge(component, var_node)
                for target in targets:
                    G.add_edge(var_node, target)

        # Calculate positions
        pos = nx.spring_layout(G)

        # Create node traces for components and variables
        component_trace = self._create_node_trace([
            (node, pos[node]) for node, attr in G.nodes(data=True)
            if attr.get('type') == 'component'
        ], 'circle')

        variable_trace = self._create_node_trace([
            (node, pos[node]) for node, attr in G.nodes(data=True)
            if attr.get('type') == 'variable'
        ], 'diamond')

        # Create edge trace
        edge_trace = self._create_edge_trace(G, pos)

        # Create figure
        figure = {
            'data': [edge_trace, component_trace, variable_trace],
            'layout': go.Layout(
                title='Data Flow Diagram',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor=self.themes[self.config.theme]["background"],
                paper_bgcolor=self.themes[self.config.theme]["background"],
                font=dict(color=self.themes[self.config.theme]["text"])
            )
        }

        return figure

    def visualize_bug_impact(self, impact_data: Dict) -> Dict[str, Any]:
        """Create visualization of bug impact"""
        G = nx.DiGraph()
        
        # Add nodes for affected components
        for component in impact_data['affected_components']:
            G.add_node(component, type='affected')
        
        # Add the source bug node
        bug_location = impact_data.get('bug_location', 'Unknown')
        G.add_node(bug_location, type='bug')
        
        # Add edges based on impact paths
        for component in impact_data['affected_components']:
            G.add_edge(bug_location, component)

        # Calculate layout
        pos = nx.spring_layout(G)

        # Create traces
        bug_trace = self._create_node_trace([
            (node, pos[node]) for node, attr in G.nodes(data=True)
            if attr.get('type') == 'bug'
        ], 'x', self.themes[self.config.theme]["highlight"])

        affected_trace = self._create_node_trace([
            (node, pos[node]) for node, attr in G.nodes(data=True)
            if attr.get('type') == 'affected'
        ], 'circle')

        edge_trace = self._create_edge_trace(G, pos)

        # Create figure
        figure = {
            'data': [edge_trace, bug_trace, affected_trace],
            'layout': go.Layout(
                title='Bug Impact Analysis',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor=self.themes[self.config.theme]["background"],
                paper_bgcolor=self.themes[self.config.theme]["background"],
                font=dict(color=self.themes[self.config.theme]["text"])
            )
        }

        return figure

    def _create_node_trace(self, nodes, symbol, color=None):
        """Helper to create node traces"""
        x = []
        y = []
        text = []
        
        for node, pos in nodes:
            x.append(pos[0])
            y.append(pos[1])
            text.append(node)

        return go.Scatter(
            x=x, y=y,
            mode='markers+text' if self.config.show_labels else 'markers',
            hoverinfo='text',
            text=text,
            textposition="bottom center",
            marker={
                'size': self.config.node_size,
                'symbol': symbol,
                'color': color or self.themes[self.config.theme]["nodes"],
                'line_width': 2
            }
        )

    def _create_edge_trace(self, G, pos):
        """Helper to create edge traces"""
        edge_x = []
        edge_y = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        return go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(
                width=self.config.edge_width,
                color=self.themes[self.config.theme]["edges"]
            ),
            hoverinfo='none',
            mode='lines'
        )

    def export_to_html(self, figure: Dict[str, Any], filename: str):
        """Export visualization to standalone HTML file"""
        try:
            fig = go.Figure(figure)
            fig.write_html(filename)
        except Exception as e:
            logger.error(f"Error exporting visualization: {str(e)}")
            raise

    def get_theme_colors(self) -> Dict[str, str]:
        """Get current theme colors"""
        return self.themes[self.config.theme]