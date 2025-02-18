from src.CRMS_Continuous_Hydrographic2subsets import continuous_subcommand
from src.CRMS_Discrete_Hydrographic2subsets import discrete_subcommand
from src.CRMS2Resample import resample_subcommand
from src.CRMS2Plot import plot_subcommand
from src.CRMS2Interpolate import interpolate_subcommand
from src.CRMS_general_functions import *

# from CRMS_Continuous_Hydrographic2subsets import continuous_subcommand
# from CRMS_Discrete_Hydrographic2subsets import discrete_subcommand
# from CRMS2Resample import resample_subcommand
# from CRMS2Plot import plot_subcommand
# from CRMS_general_functions import *

########################################################################################################################
### Cautions
########################################################################################################################

"""
click.option argument (--abcd) must be lowercase
"""


@click.group()
def click_main():
    """CRMS2Map Command Line Interface"""
    pass


# Add subcommands to the group
click_main.add_command(continuous_subcommand, name="continuous")
click_main.add_command(discrete_subcommand, name="discrete")
click_main.add_command(resample_subcommand, name="resample")
click_main.add_command(plot_subcommand, name="plot")
click_main.add_command(interpolate_subcommand, name="interpolate")

if __name__ == "__main__":
    click_main()
