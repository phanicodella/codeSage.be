# Path: codeSage.be/src/services/response_formatter.py

from typing import Any, Dict, List, Union, Optional
import json
from datetime import datetime
import logging
from pathlib import Path
import html
import re
import markdown
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class ResponseType(Enum):
    """Enumeration of possible response types"""
    ANALYSIS = "analysis"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    BUG_REPORT = "bug_report"
    SUGGESTION = "suggestion"

@dataclass
class CodeLocation:
    """Data class for code location information"""
    file_path: str
    line_start: int
    line_end: Optional[int] = None
    column_start: Optional[int] = None
    column_end: Optional[int] = None

@dataclass
class CodeSnippet:
    """Data class for code snippet information"""
    code: str
    language: str
    location: Optional[CodeLocation] = None
    context: Optional[str] = None

@dataclass
class AnalysisResult:
    """Data class for analysis results"""
    type: ResponseType
    message: str
    severity: Optional[int] = None
    confidence: Optional[float] = None
    snippets: Optional[List[CodeSnippet]] = None
    suggestions: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class ResponseFormatter:
    """
    Service for formatting analysis results and responses in a consistent structure
    for API consumption.
    """

    def __init__(self, 
                 include_metadata: bool = True,
                 max_snippet_length: int = 1000,
                 enable_markdown: bool = True):
        """
        Initialize the response formatter with configuration options.

        Args:
            include_metadata: Whether to include metadata in responses
            max_snippet_length: Maximum length for code snippets
            enable_markdown: Whether to enable markdown formatting
        """
        self.include_metadata = include_metadata
        self.max_snippet_length = max_snippet_length
        self.enable_markdown = enable_markdown
        self._initialize_patterns()

    def _initialize_patterns(self):
        """Initialize regex patterns for code processing"""
        self.patterns = {
            'leading_whitespace': re.compile(r'^\s+'),
            'file_path': re.compile(r'^.*?(?=:?\d+:|$)'),
            'line_number': re.compile(r':(\d+)(?::(\d+))?'),
            'markdown_code': re.compile(r'```(\w+)?\n(.*?)\n```', re.DOTALL)
        }

    def format_analysis_result(self, result: AnalysisResult) -> Dict[str, Any]:
        """
        Format an analysis result into a standardized response structure.

        Args:
            result: AnalysisResult object to format

        Returns:
            Dictionary containing formatted response
        """
        formatted = {
            'type': result.type.value,
            'message': self._format_message(result.message),
            'timestamp': datetime.utcnow().isoformat()
        }

        if result.severity is not None:
            formatted['severity'] = result.severity

        if result.confidence is not None:
            formatted['confidence'] = round(result.confidence, 3)

        if result.snippets:
            formatted['snippets'] = [
                self._format_snippet(snippet) for snippet in result.snippets
            ]

        if result.suggestions:
            formatted['suggestions'] = [
                self._format_message(suggestion) for suggestion in result.suggestions
            ]

        if self.include_metadata and result.metadata:
            formatted['metadata'] = result.metadata

        return formatted

    def _format_message(self, message: str) -> str:
        """
        Format a message with optional markdown processing.

        Args:
            message: Message to format

        Returns:
            Formatted message string
        """
        if not message:
            return ""

        if self.enable_markdown:
            try:
                # Convert markdown to HTML
                html_content = markdown.markdown(
                    message,
                    extensions=['fenced_code', 'tables', 'sane_lists']
                )
                # Clean up HTML for safety
                return html.unescape(html_content)
            except Exception as e:
                logger.warning(f"Markdown processing failed: {str(e)}")
                return message
        return message

    def _format_snippet(self, snippet: CodeSnippet) -> Dict[str, Any]:
        """
        Format a code snippet with proper escaping and truncation.

        Args:
            snippet: CodeSnippet object to format

        Returns:
            Dictionary containing formatted snippet
        """
        formatted_snippet = {
            'code': self._truncate_code(snippet.code),
            'language': snippet.language
        }

        if snippet.location:
            formatted_snippet['location'] = asdict(snippet.location)

        if snippet.context:
            formatted_snippet['context'] = snippet.context

        return formatted_snippet

    def _truncate_code(self, code: str) -> str:
        """
        Truncate code snippet if it exceeds maximum length.

        Args:
            code: Code string to truncate

        Returns:
            Truncated code string
        """
        if len(code) > self.max_snippet_length:
            truncated = code[:self.max_snippet_length]
            return f"{truncated}\n... [truncated]"
        return code

    def format_error_response(self, 
                            error: Exception,
                            include_traceback: bool = False) -> Dict[str, Any]:
        """
        Format an error response.

        Args:
            error: Exception object
            include_traceback: Whether to include traceback in response

        Returns:
            Dictionary containing formatted error response
        """
        response = {
            'type': ResponseType.ERROR.value,
            'message': str(error),
            'timestamp': datetime.utcnow().isoformat()
        }

        if include_traceback:
            import traceback
            response['traceback'] = traceback.format_exc()

        return response

    def format_batch_results(self, 
                           results: List[AnalysisResult]) -> Dict[str, Any]:
        """
        Format multiple analysis results into a single response.

        Args:
            results: List of AnalysisResult objects

        Returns:
            Dictionary containing formatted batch response
        """
        return {
            'type': 'batch_response',
            'count': len(results),
            'timestamp': datetime.utcnow().isoformat(),
            'results': [self.format_analysis_result(result) for result in results]
        }

    def parse_code_location(self, location_str: str) -> Optional[CodeLocation]:
        """
        Parse a string containing file path and line numbers into a CodeLocation object.

        Args:
            location_str: String containing location information

        Returns:
            CodeLocation object or None if parsing fails
        """
        try:
            file_path = self.patterns['file_path'].match(location_str).group(0)
            line_matches = self.patterns['line_number'].search(location_str)
            
            if line_matches:
                line_start = int(line_matches.group(1))
                column_start = int(line_matches.group(2)) if line_matches.group(2) else None
                
                return CodeLocation(
                    file_path=file_path,
                    line_start=line_start,
                    column_start=column_start
                )
            
            return CodeLocation(file_path=file_path, line_start=0)
        except Exception as e:
            logger.error(f"Error parsing code location '{location_str}': {str(e)}")
            return None

    def to_json(self, data: Any) -> str:
        """
        Convert response data to JSON string.

        Args:
            data: Data to convert to JSON

        Returns:
            JSON string representation of data
        """
        try:
            return json.dumps(data, default=str, ensure_ascii=False)
        except Exception as e:
            logger.error(f"JSON serialization failed: {str(e)}")
            return json.dumps({
                'type': ResponseType.ERROR.value,
                'message': 'JSON serialization failed',
                'timestamp': datetime.utcnow().isoformat()
            })