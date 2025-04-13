"""Updates icons and logos to most recent version."""
import sys
import os
from PIL import Image

version=sys.argv[1]

out_path = "docs/_static"

svg_filename = "logo.svg"
png_filename = "logo.png"
ico_filename = "favicon.ico"


out_filepath = os.path.join(os.getcwd(),
                      out_path,
                      svg_filename)

original_svg=open(os.path.join(os.getcwd(),
                               'icon_dev\\sparrowpy_transparent.svg'),
                  'r')

out=open(os.path.join(os.getcwd(),
                      out_path,
                      svg_filename),
         'w')



svg_string = original_svg.read()
svg_string = svg_string.replace('>[stand-in]<',f'>{version}<')
out.write(svg_string)

img = Image.open(os.path.join(os.getcwd(),
                      "icon_dev",
                      "sparrowpy_filled.png",
                      ))

img.save(os.path.join(os.getcwd(),
                      out_path,
                      ico_filename,
                      ))


