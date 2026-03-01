# %%
from pathlib import Path
from kapoorlabs_lightning.utils import  plot_npz_files_interactive
import os


# %%
npz_directory_current=Path('/projects/extern/nhr/nhr_ni/nhr_ni_test/nhr_ni_test_27040/dir.project/oneat_mitosis_model_adam/')
npz_directory_backup=Path('/projects/extern/nhr/nhr_ni/nhr_ni_test/nhr_ni_test_27040/dir.project/oneat_mitosis_model_adam/backup/')

npz_file_current = [npz_directory_current/file for file in os.listdir(npz_directory_current) if file.endswith('.npz')][-1]
#npz_file_backup = [npz_directory_backup/file for file in os.listdir(npz_directory_backup) if file.endswith('.npz')]

filepaths =  [npz_file_current] #+  [npz_file for npz_file in npz_file_backup] 

# %%

plot_npz_files_interactive(filepaths, save_plots=True, page_output_dir=npz_directory_current.stem)

# %%



