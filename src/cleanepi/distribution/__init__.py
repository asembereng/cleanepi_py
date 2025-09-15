"""
PyInstaller distribution creation for CleanEPI.

This module provides functionality to create standalone executable distributions
that bundle all dependencies for offline installation.
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any
import tempfile
import json
from loguru import logger

try:
    import PyInstaller
    from PyInstaller import __main__ as pyi_main
    PYINSTALLER_AVAILABLE = True
except ImportError:
    PYINSTALLER_AVAILABLE = False


class DistributionBuilder:
    """Builds standalone distributions for CleanEPI."""
    
    def __init__(self, source_dir: str = None):
        """Initialize distribution builder."""
        if source_dir is None:
            # Try to find the source directory
            self.source_dir = Path(__file__).parent.parent.parent.parent
        else:
            self.source_dir = Path(source_dir)
        
        self.system = platform.system().lower()
        self.arch = platform.machine().lower()
        self.dist_dir = self.source_dir / "dist"
        self.build_dir = self.source_dir / "build" 
        
    def check_dependencies(self) -> bool:
        """Check if required build tools are available."""
        if not PYINSTALLER_AVAILABLE:
            logger.error("PyInstaller not available. Install with: pip install pyinstaller")
            return False
        
        # Check for platform-specific build tools
        if self.system == "windows":
            try:
                subprocess.run(["makensis", "/VERSION"], capture_output=True, check=True)
                logger.info("NSIS installer found")
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning("NSIS not found - Windows installer will not be created")
        
        return True
    
    def create_spec_file(self) -> str:
        """Create PyInstaller spec file."""
        spec_content = self._generate_spec_content()
        spec_file = self.source_dir / "cleanepi.spec"
        
        with open(spec_file, 'w') as f:
            f.write(spec_content)
        
        logger.info(f"Created spec file: {spec_file}")
        return str(spec_file)
    
    def build_executable(self, console: bool = True) -> bool:
        """Build standalone executable using PyInstaller."""
        if not self.check_dependencies():
            return False
        
        spec_file = self.create_spec_file()
        
        try:
            # Run PyInstaller
            args = [
                spec_file,
                "--clean",
                "--noconfirm"
            ]
            
            if not console:
                args.append("--noconsole")
            
            logger.info("Building executable with PyInstaller...")
            pyi_main.run(args)
            
            logger.info("Executable built successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build executable: {e}")
            return False
    
    def create_installer(self) -> bool:
        """Create platform-specific installer."""
        if self.system == "windows":
            return self._create_windows_installer()
        elif self.system == "darwin":
            return self._create_macos_installer()
        elif self.system == "linux":
            return self._create_linux_packages()
        else:
            logger.error(f"Unsupported platform for installer: {self.system}")
            return False
    
    def bundle_offline_dependencies(self) -> bool:
        """Bundle all Python dependencies for offline installation."""
        try:
            # Create wheels directory
            wheels_dir = self.dist_dir / "wheels"
            wheels_dir.mkdir(parents=True, exist_ok=True)
            
            # Download all dependencies as wheels
            logger.info("Downloading dependencies as wheels...")
            subprocess.run([
                sys.executable, "-m", "pip", "wheel",
                "--wheel-dir", str(wheels_dir),
                "--no-deps",
                str(self.source_dir)
            ], check=True)
            
            # Download indirect dependencies
            subprocess.run([
                sys.executable, "-m", "pip", "wheel", 
                "--wheel-dir", str(wheels_dir),
                "pandas>=2.0.0",
                "numpy>=1.24.0", 
                "python-dateutil>=2.8.0",
                "pydantic>=2.0.0",
                "loguru>=0.7.0",
                "typing-extensions>=4.5.0",
                "fastapi>=0.100.0",
                "uvicorn>=0.22.0",
                "pydantic-settings>=2.0.0"
            ], check=True)
            
            logger.info(f"Dependencies bundled in {wheels_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to bundle dependencies: {e}")
            return False
    
    def create_offline_installer(self) -> str:
        """Create complete offline installer package."""
        # Build executable
        if not self.build_executable():
            raise RuntimeError("Failed to build executable")
        
        # Bundle dependencies
        if not self.bundle_offline_dependencies():
            raise RuntimeError("Failed to bundle dependencies")
        
        # Create installer package directory
        installer_dir = self.dist_dir / f"cleanepi-installer-{self.system}-{self.arch}"
        installer_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy executable
        exe_name = "cleanepi.exe" if self.system == "windows" else "cleanepi"
        exe_source = self.dist_dir / "cleanepi" / exe_name
        if exe_source.exists():
            shutil.copy2(exe_source, installer_dir / exe_name)
        
        # Copy wheels
        wheels_source = self.dist_dir / "wheels"
        if wheels_source.exists():
            shutil.copytree(wheels_source, installer_dir / "wheels", dirs_exist_ok=True)
        
        # Copy web assets
        web_assets = self.source_dir / "src" / "cleanepi" / "web"
        if web_assets.exists():
            shutil.copytree(web_assets, installer_dir / "web", dirs_exist_ok=True)
        
        # Create installation script
        self._create_install_script(installer_dir)
        
        # Create archive
        archive_name = f"cleanepi-{self.system}-{self.arch}"
        archive_path = shutil.make_archive(
            str(self.dist_dir / archive_name),
            'zip' if self.system == 'windows' else 'gztar',
            str(installer_dir)
        )
        
        logger.info(f"Created offline installer: {archive_path}")
        return archive_path
    
    def _generate_spec_content(self) -> str:
        """Generate PyInstaller spec file content."""
        return f"""
# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

# Add source directory to path
sys.path.insert(0, str(Path.cwd() / "src"))

a = Analysis(
    ['src/cleanepi/cli.py'],
    pathex=[str(Path.cwd())],
    binaries=[],
    datas=[
        ('src/cleanepi/web/static', 'cleanepi/web/static'),
        ('src/cleanepi/web/templates', 'cleanepi/web/templates'),
    ],
    hiddenimports=[
        'cleanepi.service.manager',
        'cleanepi.service.web_server',
        'cleanepi.web.api',
        'cleanepi.web.jobs',
        'uvicorn.lifespan.on',
        'uvicorn.lifespan.off',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets.websockets_impl',
        'uvicorn.protocols.http.httptools_impl',
        'uvicorn.protocols.http.h11_impl',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='cleanepi',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# For macOS, create an app bundle
{'app = BUNDLE(exe, name="CleanEPI.app", icon=None, bundle_identifier="org.cleanepi.app")' if platform.system() == 'Darwin' else ''}
"""
    
    def _create_install_script(self, installer_dir: Path):
        """Create installation script."""
        if self.system == "windows":
            script_content = """@echo off
echo Installing CleanEPI...

REM Install Python dependencies
echo Installing dependencies...
python -m pip install --find-links wheels --no-index --force-reinstall cleanepi-python

REM Install as service
echo Setting up service...
cleanepi service install --host 127.0.0.1 --port 8000

echo Installation complete!
echo You can now use:
echo   cleanepi --help            - Show command help
echo   cleanepi web               - Start web interface
echo   cleanepi service status    - Check service status

pause
"""
            script_file = installer_dir / "install.bat"
        else:
            script_content = """#!/bin/bash
echo "Installing CleanEPI..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found"
    exit 1
fi

# Install Python dependencies
echo "Installing dependencies..."
python3 -m pip install --find-links wheels --no-index --force-reinstall cleanepi-python

# Install as service
echo "Setting up service..."
./cleanepi service install --host 127.0.0.1 --port 8000

echo "Installation complete!"
echo "You can now use:"
echo "  ./cleanepi --help            - Show command help"
echo "  ./cleanepi web               - Start web interface"
echo "  ./cleanepi service status    - Check service status"
"""
            script_file = installer_dir / "install.sh"
            # Make executable
            script_file.chmod(0o755)
        
        script_file.write_text(script_content)
        logger.info(f"Created install script: {script_file}")
    
    def _create_windows_installer(self) -> bool:
        """Create Windows NSIS installer."""
        try:
            nsis_script = self._generate_nsis_script()
            nsis_file = self.dist_dir / "cleanepi_installer.nsi"
            
            with open(nsis_file, 'w') as f:
                f.write(nsis_script)
            
            # Run NSIS
            subprocess.run(["makensis", str(nsis_file)], check=True)
            
            logger.info("Windows installer created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Windows installer: {e}")
            return False
    
    def _create_macos_installer(self) -> bool:
        """Create macOS PKG installer."""
        try:
            # Create application bundle structure
            app_dir = self.dist_dir / "CleanEPI.app"
            contents_dir = app_dir / "Contents"
            macos_dir = contents_dir / "MacOS"
            resources_dir = contents_dir / "Resources"
            
            # Create directories
            for dir_path in [app_dir, contents_dir, macos_dir, resources_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Copy executable
            exe_source = self.dist_dir / "cleanepi" / "cleanepi"
            if exe_source.exists():
                shutil.copy2(exe_source, macos_dir / "cleanepi")
            
            # Create Info.plist
            plist_content = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDisplayName</key>
    <string>CleanEPI</string>
    <key>CFBundleIdentifier</key>
    <string>org.cleanepi.app</string>
    <key>CFBundleName</key>
    <string>CleanEPI</string>
    <key>CFBundleVersion</key>
    <string>0.1.0</string>
    <key>CFBundleExecutable</key>
    <string>cleanepi</string>
</dict>
</plist>"""
            
            (contents_dir / "Info.plist").write_text(plist_content)
            
            logger.info("macOS app bundle created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create macOS installer: {e}")
            return False
    
    def _create_linux_packages(self) -> bool:
        """Create Linux DEB and RPM packages."""
        try:
            # Create DEB package structure
            deb_dir = self.dist_dir / "cleanepi-deb"
            debian_dir = deb_dir / "DEBIAN"
            usr_bin_dir = deb_dir / "usr" / "bin"
            
            # Create directories
            for dir_path in [deb_dir, debian_dir, usr_bin_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Copy executable
            exe_source = self.dist_dir / "cleanepi" / "cleanepi"
            if exe_source.exists():
                shutil.copy2(exe_source, usr_bin_dir / "cleanepi")
                (usr_bin_dir / "cleanepi").chmod(0o755)
            
            # Create control file
            control_content = """Package: cleanepi
Version: 0.1.0
Section: science
Priority: optional
Architecture: amd64
Maintainer: CleanEPI Team <maintainer@cleanepi.org>
Description: Clean and standardize epidemiological data
 CleanEPI is a comprehensive tool for cleaning and standardizing
 epidemiological data with both command-line and web interfaces.
"""
            
            (debian_dir / "control").write_text(control_content)
            
            # Build DEB package
            subprocess.run([
                "dpkg-deb", "--build", str(deb_dir), 
                str(self.dist_dir / "cleanepi_0.1.0_amd64.deb")
            ], check=True)
            
            logger.info("Linux DEB package created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Linux packages: {e}")
            return False
    
    def _generate_nsis_script(self) -> str:
        """Generate NSIS installer script for Windows."""
        return """
!define APPNAME "CleanEPI"
!define APPVERSION "0.1.0"
!define APPDIR "CleanEPI"

Name "${APPNAME}"
OutFile "CleanEPI_Setup.exe"
InstallDir "$PROGRAMFILES\\${APPDIR}"

Page directory
Page instfiles

Section "Main Section" SecMain
    SetOutPath "$INSTDIR"
    File /r "dist\\cleanepi\\*.*"
    
    # Create uninstaller
    WriteUninstaller "$INSTDIR\\Uninstall.exe"
    
    # Add to Add/Remove Programs
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "DisplayName" "${APPNAME}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "UninstallString" "$INSTDIR\\Uninstall.exe"
    
    # Create start menu shortcuts
    CreateDirectory "$SMPROGRAMS\\${APPNAME}"
    CreateShortCut "$SMPROGRAMS\\${APPNAME}\\${APPNAME}.lnk" "$INSTDIR\\cleanepi.exe"
    CreateShortCut "$SMPROGRAMS\\${APPNAME}\\Uninstall.lnk" "$INSTDIR\\Uninstall.exe"
    
SectionEnd

Section "Uninstall"
    Delete "$INSTDIR\\*.*"
    RMDir /r "$INSTDIR"
    
    Delete "$SMPROGRAMS\\${APPNAME}\\*.*"
    RMDir "$SMPROGRAMS\\${APPNAME}"
    
    DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}"
SectionEnd
"""


def main():
    """Main function for distribution building."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build CleanEPI distributions")
    parser.add_argument("--type", choices=["executable", "installer", "offline"], 
                       default="offline", help="Type of distribution to build")
    parser.add_argument("--source-dir", help="Source directory path")
    
    args = parser.parse_args()
    
    builder = DistributionBuilder(args.source_dir)
    
    if args.type == "executable":
        success = builder.build_executable()
    elif args.type == "installer":
        success = builder.create_installer()
    elif args.type == "offline":
        try:
            archive_path = builder.create_offline_installer()
            print(f"Offline installer created: {archive_path}")
            success = True
        except Exception as e:
            logger.error(f"Failed to create offline installer: {e}")
            success = False
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()