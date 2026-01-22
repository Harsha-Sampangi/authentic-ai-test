
import os
import uuid
import logging
from pathlib import Path
import yt_dlp

logger = logging.getLogger(__name__)

class MediaDownloader:
    """
    Downloads media (video/audio) from URLs using yt-dlp
    """
    
    def __init__(self, upload_dir):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        
    def download(self, url):
        """
        Download media from URL
        
        Args:
            url: URL string (Youtube, Instagram, etc.)
            
        Returns:
            dict: {
                'path': Path object to downloaded file,
                'filename': str,
                'type': 'video' or 'audio',
                'title': str
            }
        """
        file_id = str(uuid.uuid4())
        # Template for output filename: UUID.extension
        output_template = str(self.upload_dir / f"{file_id}.%(ext)s")
        
        ydl_opts = {
            'outtmpl': output_template,
            'format': 'best[ext=mp4]/best',  # Avoid merging to prevent ffmpeg error
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
            'max_filesize': 100 * 1024 * 1024, # 100 MB limit
        }
        
        try:
            logger.info(f"⬇️ Downloading from URL: {url}")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info first to get metadata
                info = ydl.extract_info(url, download=True)
                
                # Get the filename that was generated
                filename = ydl.prepare_filename(info)
                
                # Verify file exists (extract_info returns expected filename, but verify actual)
                downloaded_path = Path(filename)
                
                if not downloaded_path.exists():
                     # Sometimes extension differs, look for file_id
                     for f in self.upload_dir.glob(f"{file_id}.*"):
                         downloaded_path = f
                         break
                
                if not downloaded_path.exists():
                    raise Exception("File not found after download")
                
                # Determine type
                # Simple check based on extension or info
                ext = downloaded_path.suffix.lower()
                media_type = 'video'
                if ext in ['.mp3', '.wav', '.opus', '.m4a', '.flac', '.aac']:
                    media_type = 'audio'
                
                return {
                    'path': downloaded_path,
                    'filename': downloaded_path.name,
                    'type': media_type,
                    'title': info.get('title', 'Downloaded Media')
                }
                
        except Exception as e:
            logger.error(f"Download error: {str(e)}")
            raise e
