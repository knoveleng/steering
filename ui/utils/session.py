"""
Session management utilities
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import json
from datetime import datetime


class SessionManager:
    """Manage calibration sessions"""
    
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
    
    def list_sessions(self, model_name: Optional[str] = None) -> List[Dict]:
        """
        List available calibration sessions
        
        Args:
            model_name: Optional model name to filter
            
        Returns:
            List of session info dictionaries
        """
        if not self.artifacts_dir.exists():
            return []
        
        pattern = "calibration_*"
        if model_name:
            pattern = f"calibration_{model_name}_*"
        
        sessions = []
        for session_dir in self.artifacts_dir.glob(pattern):
            if session_dir.is_dir():
                info = self.get_session_info(session_dir)
                sessions.append(info)
        
        # Sort by creation time (newest first)
        sessions.sort(key=lambda x: x['modified'], reverse=True)
        
        return sessions
    
    def get_session_info(self, session_path: Path) -> Dict:
        """
        Get information about a session
        
        Args:
            session_path: Path to session directory
            
        Returns:
            Dictionary with session information
        """
        info = {
            'path': str(session_path),
            'name': session_path.name,
            'modified': session_path.stat().st_mtime,
            'modified_str': datetime.fromtimestamp(
                session_path.stat().st_mtime
            ).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Try to load config
        config_file = session_path / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                info['model'] = config.get('model', {}).get('name', 'unknown')
                info['mode'] = config.get('steering', {}).get('mode', 'standard')
            except:
                pass
        
        # Check for required files
        required_files = ['plane.pt', 'config.json']
        info['complete'] = all(
            (session_path / f).exists() for f in required_files
        )
        
        return info
    
    def validate_session(self, session_path: Path) -> bool:
        """
        Validate that a session has all required files
        
        Args:
            session_path: Path to session directory
            
        Returns:
            True if valid, False otherwise
        """
        required_files = ['plane.pt', 'config.json']
        return all((session_path / f).exists() for f in required_files)