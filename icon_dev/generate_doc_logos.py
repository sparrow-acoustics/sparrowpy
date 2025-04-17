"""Updates icons and logos to most recent version."""
import os
import re
import shutil
import sparrowpy

def update_all_icons(base_path: str):
    """Updates sparrowpy version in logo and generates icon."""

    out_path = os.path.join(base_path,"docs/_static")
    in_path = os.path.join(base_path,"icon_dev")


    versioned  = set_svg_version(
                    template_filepath=os.path.join(in_path,"logo.svg"),
                    version=sparrowpy.__version__,
                    out_filename="_temp_ver.svg")

    no_version = remove_versioning_layer(
                    template_filepath=os.path.join(in_path,"logo.svg"),
                    out_filename="_temp.svg")

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

def remove_versioning_layer(template_filepath: str,
                    out_filename="temp.svg"):
    """Read template .svg and remove version banner."""

    out_filepath = os.path.join(
                    os.path.split(template_filepath)[0],
                    out_filename)

    template_file = open(template_filepath, 'r')
    out_file=open(out_filepath, 'w')

    # read template svg as string
    template_string = template_file.read()

    # replace standin version number
    out_string = re.sub(r'height=".+?mm"', 'height="100mm"',
                        template_string, count=1)
    out_string = re.sub(r'viewBox=".+?"', 'viewBox="0 0 100 100"',
                        out_string)
    out_string = out_string.replace('id="layer1"',
                                    'id="layer1" visibility="hidden"')

    # write logo svg
    out_file.write(out_string)

    return out_filepath

if __name__=="__main__":
    update_all_icons(base_path=os.getcwd())


