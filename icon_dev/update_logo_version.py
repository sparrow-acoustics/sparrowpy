"""Updates icons and logos to most recent version."""
import os
from PIL import Image
import shutil
import sparrowpy

def update_all_icons(base_path: str):
    """Updates sparrowpy version in logo and generates icon."""

    out_path = os.path.join(base_path,"docs/_static")
    in_path = os.path.join(base_path,"icon_dev")

    icon = generate_icon_from_png(
                template_filepath=os.path.join(in_path,
                                               "logo_no_version.png"))

    no_version = set_svg_version(
                    template_filepath=os.path.join(in_path,"logo.svg"),
                    out_filename="temp.svg")

    versioned  = set_svg_version(
                    template_filepath=os.path.join(in_path,"logo.svg"),
                    version=sparrowpy.__version__,
                    out_filename="temp_ver.svg")

    copy_to_destination(in_filepath=icon,
                        destination_dir=out_path,
                        destination_name="favicon.ico")
    copy_to_destination(in_filepath=no_version,
                        destination_dir=out_path,
                        destination_name="logo_nover.svg")
    copy_to_destination(in_filepath=versioned,
                        destination_dir=out_path,
                        destination_name="logo.svg")

def copy_to_destination(in_filepath: str,
                        destination_dir: str,
                        destination_name=None):
    """Copy file to given destination."""

    if destination_name is None:
        destination_name = os.path.split(in_filepath)[-1]

    shutil.copyfile(in_filepath,
                    os.path.join(destination_dir,
                                 destination_name),
                    )

def set_svg_version(template_filepath: str,
                    version="",
                    out_filename="temp.svg"):
    """Read template .svg and update stand in text with version."""

    out_filepath = os.path.join(
                    os.path.split(template_filepath)[0],
                    out_filename)

    template_file = open(template_filepath, 'r')
    out_file=open(out_filepath, 'w')

    # read template svg as string
    template_string = template_file.read()

    # replace standin version number
    out_string = template_string.replace('>[stand-in]<',
                                         f'>{version}<')

    # write logo svg
    out_file.write(out_string)

    return out_filepath




def generate_icon_from_png(template_filepath: str,
                           out_filename = "temp.ico"):
    """Generate .ico image from .png input."""
    img = Image.open(template_filepath)

    out_fpath = os.path.join(
                    os.path.split(template_filepath)[0],
                    out_filename)

    img.save(out_fpath)

    return out_fpath


if __name__=="__main__":
    update_all_icons(base_path=os.getcwd())
