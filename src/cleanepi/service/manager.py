"""
Cross-platform service management for cleanepi web application.
"""

import os
import sys
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
import json
from loguru import logger

from ..core.config import WebConfig


class ServiceManager:
    """Manages cleanepi web application as a system service."""
    
    def __init__(self, config: Optional[WebConfig] = None):
        """Initialize service manager."""
        self.config = config or WebConfig()
        self.system = platform.system().lower()
        self.service_name = "cleanepi-web"
        self.app_name = "CleanEPI Web Service"
        
    def install_service(self, 
                       host: str = "127.0.0.1", 
                       port: int = 8000,
                       auto_start: bool = True) -> bool:
        """Install cleanepi as a system service."""
        try:
            if self.system == "linux":
                return self._install_systemd_service(host, port, auto_start)
            elif self.system == "windows":
                return self._install_windows_service(host, port, auto_start)
            elif self.system == "darwin":
                return self._install_macos_service(host, port, auto_start)
            else:
                logger.error(f"Unsupported operating system: {self.system}")
                return False
        except Exception as e:
            logger.error(f"Failed to install service: {e}")
            return False
    
    def uninstall_service(self) -> bool:
        """Uninstall cleanepi system service."""
        try:
            if self.system == "linux":
                return self._uninstall_systemd_service()
            elif self.system == "windows":
                return self._uninstall_windows_service()
            elif self.system == "darwin":
                return self._uninstall_macos_service()
            else:
                logger.error(f"Unsupported operating system: {self.system}")
                return False
        except Exception as e:
            logger.error(f"Failed to uninstall service: {e}")
            return False
    
    def start_service(self) -> bool:
        """Start the cleanepi service."""
        try:
            if self.system == "linux":
                result = subprocess.run(
                    ["sudo", "systemctl", "start", self.service_name],
                    capture_output=True, text=True
                )
                return result.returncode == 0
            elif self.system == "windows":
                result = subprocess.run(
                    ["sc", "start", self.service_name],
                    capture_output=True, text=True
                )
                return result.returncode == 0
            elif self.system == "darwin":
                result = subprocess.run(
                    ["launchctl", "start", f"local.{self.service_name}"],
                    capture_output=True, text=True
                )
                return result.returncode == 0
            return False
        except Exception as e:
            logger.error(f"Failed to start service: {e}")
            return False
    
    def stop_service(self) -> bool:
        """Stop the cleanepi service."""
        try:
            if self.system == "linux":
                result = subprocess.run(
                    ["sudo", "systemctl", "stop", self.service_name],
                    capture_output=True, text=True
                )
                return result.returncode == 0
            elif self.system == "windows":
                result = subprocess.run(
                    ["sc", "stop", self.service_name],
                    capture_output=True, text=True
                )
                return result.returncode == 0
            elif self.system == "darwin":
                result = subprocess.run(
                    ["launchctl", "stop", f"local.{self.service_name}"],
                    capture_output=True, text=True
                )
                return result.returncode == 0
            return False
        except Exception as e:
            logger.error(f"Failed to stop service: {e}")
            return False
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status."""
        try:
            if self.system == "linux":
                result = subprocess.run(
                    ["systemctl", "is-active", self.service_name],
                    capture_output=True, text=True
                )
                active = result.stdout.strip() == "active"
                
                result = subprocess.run(
                    ["systemctl", "is-enabled", self.service_name],
                    capture_output=True, text=True
                )
                enabled = result.stdout.strip() == "enabled"
                
                return {
                    "installed": True,
                    "running": active,
                    "enabled": enabled,
                    "system": self.system
                }
            elif self.system == "windows":
                result = subprocess.run(
                    ["sc", "query", self.service_name],
                    capture_output=True, text=True
                )
                installed = result.returncode == 0
                running = "RUNNING" in result.stdout if installed else False
                
                return {
                    "installed": installed,
                    "running": running,
                    "enabled": installed,  # Windows services are typically enabled if installed
                    "system": self.system
                }
            elif self.system == "darwin":
                result = subprocess.run(
                    ["launchctl", "print", f"gui/{os.getuid()}/local.{self.service_name}"],
                    capture_output=True, text=True
                )
                installed = result.returncode == 0
                running = "state = running" in result.stdout if installed else False
                
                return {
                    "installed": installed,
                    "running": running,
                    "enabled": installed,
                    "system": self.system
                }
            
            return {
                "installed": False,
                "running": False,
                "enabled": False,
                "system": self.system
            }
        except Exception as e:
            logger.error(f"Failed to get service status: {e}")
            return {
                "installed": False,
                "running": False,
                "enabled": False,
                "system": self.system,
                "error": str(e)
            }
    
    def _install_systemd_service(self, host: str, port: int, auto_start: bool) -> bool:
        """Install systemd service on Linux."""
        service_content = self._generate_systemd_service(host, port)
        service_file = f"/etc/systemd/system/{self.service_name}.service"
        
        # Write service file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.service', delete=False) as f:
            f.write(service_content)
            temp_file = f.name
        
        try:
            # Copy to systemd directory
            subprocess.run(
                ["sudo", "cp", temp_file, service_file],
                check=True
            )
            
            # Reload systemd
            subprocess.run(
                ["sudo", "systemctl", "daemon-reload"],
                check=True
            )
            
            if auto_start:
                # Enable service
                subprocess.run(
                    ["sudo", "systemctl", "enable", self.service_name],
                    check=True
                )
                
                # Start service
                subprocess.run(
                    ["sudo", "systemctl", "start", self.service_name],
                    check=True
                )
            
            logger.info(f"Successfully installed systemd service: {self.service_name}")
            return True
            
        finally:
            os.unlink(temp_file)
    
    def _install_windows_service(self, host: str, port: int, auto_start: bool) -> bool:
        """Install Windows service."""
        # For Windows, we'll create a batch script and use nssm or sc command
        python_exe = sys.executable
        script_content = self._generate_windows_script(host, port)
        
        # Create script file
        app_dir = Path.home() / "AppData" / "Local" / "CleanEPI"
        app_dir.mkdir(parents=True, exist_ok=True)
        script_file = app_dir / "cleanepi_service.bat"
        
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        try:
            # Create Windows service using sc command
            cmd = [
                "sc", "create", self.service_name,
                "binPath=", str(script_file),
                "DisplayName=", self.app_name,
                "start=", "auto" if auto_start else "demand"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Successfully installed Windows service: {self.service_name}")
                if auto_start:
                    self.start_service()
                return True
            else:
                logger.error(f"Failed to create Windows service: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to install Windows service: {e}")
            return False
    
    def _install_macos_service(self, host: str, port: int, auto_start: bool) -> bool:
        """Install macOS LaunchAgent."""
        plist_content = self._generate_macos_plist(host, port)
        
        # Create LaunchAgents directory
        launch_agents_dir = Path.home() / "Library" / "LaunchAgents"
        launch_agents_dir.mkdir(parents=True, exist_ok=True)
        
        plist_file = launch_agents_dir / f"local.{self.service_name}.plist"
        
        try:
            with open(plist_file, 'w') as f:
                f.write(plist_content)
            
            if auto_start:
                # Load the service
                subprocess.run(
                    ["launchctl", "load", str(plist_file)],
                    check=True
                )
            
            logger.info(f"Successfully installed macOS LaunchAgent: {self.service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to install macOS service: {e}")
            return False
    
    def _generate_systemd_service(self, host: str, port: int) -> str:
        """Generate systemd service file content."""
        python_exe = sys.executable
        cleanepi_path = subprocess.run(
            [python_exe, "-c", "import cleanepi; print(cleanepi.__file__)"],
            capture_output=True, text=True
        ).stdout.strip()
        
        return f"""[Unit]
Description={self.app_name}
After=network.target

[Service]
Type=exec
User=cleanepi
Group=cleanepi
WorkingDirectory=/opt/cleanepi
ExecStart={python_exe} -m cleanepi.service.web_server --host {host} --port {port}
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
    
    def _generate_windows_script(self, host: str, port: int) -> str:
        """Generate Windows batch script."""
        python_exe = sys.executable
        return f"""@echo off
cd /d "%~dp0"
"{python_exe}" -m cleanepi.service.web_server --host {host} --port {port}
"""
    
    def _generate_macos_plist(self, host: str, port: int) -> str:
        """Generate macOS plist file content."""
        python_exe = sys.executable
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>local.{self.service_name}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_exe}</string>
        <string>-m</string>
        <string>cleanepi.service.web_server</string>
        <string>--host</string>
        <string>{host}</string>
        <string>--port</string>
        <string>{port}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardErrorPath</key>
    <string>/tmp/cleanepi.err</string>
    <key>StandardOutPath</key>
    <string>/tmp/cleanepi.out</string>
</dict>
</plist>
"""
    
    def _uninstall_systemd_service(self) -> bool:
        """Uninstall systemd service."""
        try:
            # Stop service
            subprocess.run(
                ["sudo", "systemctl", "stop", self.service_name],
                capture_output=True
            )
            
            # Disable service
            subprocess.run(
                ["sudo", "systemctl", "disable", self.service_name],
                capture_output=True
            )
            
            # Remove service file
            service_file = f"/etc/systemd/system/{self.service_name}.service"
            subprocess.run(
                ["sudo", "rm", "-f", service_file],
                check=True
            )
            
            # Reload systemd
            subprocess.run(
                ["sudo", "systemctl", "daemon-reload"],
                check=True
            )
            
            logger.info(f"Successfully uninstalled systemd service: {self.service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to uninstall systemd service: {e}")
            return False
    
    def _uninstall_windows_service(self) -> bool:
        """Uninstall Windows service."""
        try:
            # Stop service
            subprocess.run(
                ["sc", "stop", self.service_name],
                capture_output=True
            )
            
            # Delete service
            result = subprocess.run(
                ["sc", "delete", self.service_name],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully uninstalled Windows service: {self.service_name}")
                return True
            else:
                logger.error(f"Failed to delete Windows service: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to uninstall Windows service: {e}")
            return False
    
    def _uninstall_macos_service(self) -> bool:
        """Uninstall macOS LaunchAgent."""
        try:
            plist_file = Path.home() / "Library" / "LaunchAgents" / f"local.{self.service_name}.plist"
            
            # Unload service
            subprocess.run(
                ["launchctl", "unload", str(plist_file)],
                capture_output=True
            )
            
            # Remove plist file
            if plist_file.exists():
                plist_file.unlink()
            
            logger.info(f"Successfully uninstalled macOS LaunchAgent: {self.service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to uninstall macOS service: {e}")
            return False