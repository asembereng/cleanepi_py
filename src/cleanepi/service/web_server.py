"""
Web server manager for running cleanepi web application.
"""

import argparse
import webbrowser
import signal
import sys
import time
from typing import Optional
import uvicorn
from loguru import logger

from ..web.api import CleaningAPI
from ..core.config import WebConfig


class WebServerManager:
    """Manages the cleanepi web server."""
    
    def __init__(self, config: Optional[WebConfig] = None):
        """Initialize web server manager."""
        self.config = config or WebConfig()
        self.server = None
        self.running = False
        
    def start_server(self, 
                    host: str = "127.0.0.1", 
                    port: int = 8000,
                    open_browser: bool = True,
                    reload: bool = False) -> None:
        """Start the web server."""
        try:
            # Create FastAPI app
            api = CleaningAPI(self.config)
            app = api.app
            
            # Setup signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            logger.info(f"Starting CleanEPI web server on http://{host}:{port}")
            
            # Open browser if requested
            if open_browser:
                # Small delay to ensure server starts before opening browser
                import threading
                def delayed_browser_open():
                    time.sleep(2)
                    try:
                        webbrowser.open(f"http://{host}:{port}")
                        logger.info(f"Opened browser to http://{host}:{port}")
                    except Exception as e:
                        logger.warning(f"Could not open browser: {e}")
                
                threading.Thread(target=delayed_browser_open, daemon=True).start()
            
            self.running = True
            
            # Run the server
            uvicorn.run(
                app,
                host=host,
                port=port,
                reload=reload,
                log_level="info"
            )
            
        except Exception as e:
            logger.error(f"Failed to start web server: {e}")
            sys.exit(1)
    
    def stop_server(self) -> None:
        """Stop the web server."""
        if self.server:
            self.server.should_exit = True
        self.running = False
        logger.info("Web server stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop_server()
        sys.exit(0)


def main():
    """Main entry point for web server."""
    parser = argparse.ArgumentParser(description="CleanEPI Web Server")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open browser"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    # Create and start web server
    manager = WebServerManager()
    manager.start_server(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        reload=args.reload
    )


if __name__ == "__main__":
    main()