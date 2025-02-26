#!/usr/bin/env python3
"""
Start Script for Integrated CoD WaW Zombies Bot with Frank Castle AI
This script launches the integrated bot with command line options
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("start_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("StartBot")

def check_dependencies():
    """Check and report on dependencies"""
    missing = []
    
    try:
        import numpy
        logger.info("NumPy is installed ✓")
    except ImportError:
        missing.append("numpy")
        logger.error("NumPy is missing ✗")
    
    try:
        import cv2
        logger.info(f"OpenCV is installed ({cv2.__version__}) ✓")
    except ImportError:
        missing.append("opencv-python-headless")
        logger.error("OpenCV is missing ✗")
    
    try:
        import mss
        logger.info("MSS is installed ✓")
    except ImportError:
        missing.append("mss")
        logger.warning("MSS is missing (optional, for screen capture) ✗")
    
    # Check for platform-specific dependencies
    import platform
    
    if platform.system() == "Linux":
        try:
            from Xlib import display
            logger.info("Python-Xlib is installed ✓")
        except ImportError:
            missing.append("python-xlib")
            logger.warning("Python-Xlib is missing (needed for X11 screen capture on Linux) ✗")
    
    elif platform.system() == "Windows":
        try:
            import win32gui
            logger.info("PyWin32 is installed ✓")
        except ImportError:
            missing.append("pywin32")
            logger.warning("PyWin32 is missing (optional, for Win32 screen capture) ✗")
    
    return missing

def display_banner():
    """Display a fancy banner for the bot"""
    banner = r"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ██████╗ ██████╗ ██████╗     ██╗    ██╗ █████╗ ██╗    ██╗      ║
║  ██╔════╝██╔═══██╗██╔══██╗    ██║    ██║██╔══██╗██║    ██║      ║
║  ██║     ██║   ██║██║  ██║    ██║ █╗ ██║███████║██║ █╗ ██║      ║
║  ██║     ██║   ██║██║  ██║    ██║███╗██║██╔══██║██║███╗██║      ║
║  ╚██████╗╚██████╔╝██████╔╝    ╚███╔███╔╝██║  ██║╚███╔███╔╝      ║
║   ╚═════╝ ╚═════╝ ╚═════╝      ╚══╝╚══╝ ╚═╝  ╚═╝ ╚══╝╚══╝       ║
║                                                                  ║
║        ███████╗ ██████╗ ███╗   ███╗██████╗ ██╗███████╗          ║
║        ╚══███╔╝██╔═══██╗████╗ ████║██╔══██╗██║██╔════╝          ║
║          ███╔╝ ██║   ██║██╔████╔██║██████╔╝██║█████╗            ║
║         ███╔╝  ██║   ██║██║╚██╔╝██║██╔══██╗██║██╔══╝            ║
║        ███████╗╚██████╔╝██║ ╚═╝ ██║██████╔╝██║███████╗          ║
║        ╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚═════╝ ╚═╝╚══════╝          ║
║                                                                  ║
║   ███████╗██████╗  █████╗ ███╗   ██╗██╗  ██╗    █████╗ ██╗      ║
║   ██╔════╝██╔══██╗██╔══██╗████╗  ██║██║ ██╔╝   ██╔══██╗██║      ║
║   █████╗  ██████╔╝███████║██╔██╗ ██║█████╔╝    ███████║██║      ║
║   ██╔══╝  ██╔══██╗██╔══██║██║╚██╗██║██╔═██╗    ██╔══██║██║      ║
║   ██║     ██║  ██║██║  ██║██║ ╚████║██║  ██╗██╗██║  ██║██║      ║
║   ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    # Print banner
    print("\033[94m" + banner + "\033[0m")  # Blue color
    
    # Print subtitle
    subtitle = "Integrated Bot System with Enhanced Screen Capture"
    print("\033[93m" + "=" * len(subtitle) + "\033[0m")  # Yellow
    print("\033[93m" + subtitle + "\033[0m")
    print("\033[93m" + "=" * len(subtitle) + "\033[0m")
    print()

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Start the Integrated CoD WaW Zombies Bot")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (for development)")
    parser.add_argument("--delay", type=int, default=10, help="Startup delay in seconds")
    parser.add_argument("--capture", type=str, choices=["mss", "pil", "x11", "win32", "d3d", "auto"], 
                      default="auto", help="Preferred capture method")
    parser.add_argument("--check-only", action="store_true", help="Only check dependencies, don't start the bot")
    parser.add_argument("--test-capture", action="store_true", help="Test screen capture methods")
    args = parser.parse_args()
    
    # Display banner
    display_banner()
    
    # Ensure data directories exist
    os.makedirs("data/memory", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("config", exist_ok=True)
    
    # Check dependencies
    logger.info("Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print("\n\033[91mWARNING: Some dependencies are missing!\033[0m")
        print("You can install them with the following command:")
        print(f"\033[92mpip install {' '.join(missing)}\033[0m\n")
        
        if args.check_only:
            return
    else:
        print("\n\033[92mAll required dependencies are installed!\033[0m\n")
        
        if args.check_only:
            return
    
    # Test screen capture if requested
    if args.test_capture:
        print("\033[93mTesting screen capture methods...\033[0m")
        try:
            from enhanced_screen_capture import test_capture
            test_capture(duration=10)
            return
        except ImportError:
            logger.error("Could not import screen capture module for testing")
            return
    
    # Preferred capture method
    capture_method = None if args.capture == "auto" else args.capture
    
    # Start the bot
    try:
        # Import and run the integrated AI
        print("\033[93mStarting the Integrated Bot...\033[0m")
        
        # Build command
        cmd_args = []
        if args.headless:
            cmd_args.append("--headless")
        if args.delay != 10:
            cmd_args.append(f"--delay {args.delay}")
        if capture_method:
            cmd_args.append(f"--capture {capture_method}")
        
        cmd = f"python integrated_ai.py {' '.join(cmd_args)}"
        print(f"Command: {cmd}")
        
        # Start the bot
        from integrated_ai import IntegratedAI, main
        main()
        
    except ImportError as e:
        logger.error(f"Failed to import integrated AI: {e}")
        print("\033[91mError: Could not start the bot. Make sure all files are in place.\033[0m")
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        print(f"\033[91mError: {e}\033[0m")

if __name__ == "__main__":
    main()