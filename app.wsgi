import sys
import logging
logging.basicConfig (level=logging. DEBUG, filename='/var/www/html/IntelligWebApp/WebApp.log', format='$ (asctime)s $ (message) s')
sys.path. insert (0,"/var/www/html/IntelligWebApp')
sys.path. insert (0,"/var/www/html/IntelligWebApp/noel_env/lib/python3.10/site-packages') 
from app import app as application
