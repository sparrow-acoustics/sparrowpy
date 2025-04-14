"""Updates icons and logos to most recent version."""
import os
from PIL import Image
import sparrowpy

def main():
    """Updates sparrowpy version in logo and generates icon."""
    # TO DO split into two separate functions
    version=sparrowpy.__version__

    out_path = "docs/_static"

    svg_filename = "logo.svg"
    ico_filename = "favicon.ico"

    original_svg=open(os.path.join(os.getcwd(),
                                'icon_dev',
                                'logo.svg'),
                    'r')

    out=open(os.path.join(os.getcwd(),
                        out_path,
                        svg_filename),
            'w')

    # read template svg as string
    svg_string = original_svg.read()

    # replace standin version number
    svg_string = svg_string.replace('>[stand-in]<',
                                    f'>{version}<')

    # write logo svg
    out.write(svg_string)

    img = Image.open(os.path.join(os.getcwd(),
                        "icon_dev",
                        "logo_no_version.png",
                        ))

    img.save(os.path.join(os.getcwd(),
                        out_path,
                        ico_filename,
                        ))

if __name__=="__main__":
    main()
