# Potenetially usefull code snippets

## Create output log .txt-file

out_file = open(f"{dir_mgr.paths.this_val_out_dir}/out.txt",
                 'w', encoding="utf8")
sys.stdout = out_file
