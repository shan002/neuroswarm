# while in the behavior folder i.e. pwd is out/agg/, zip each folder
# zip folders will have the folder structure from pwd, i.e. 10/project_folder/
ls */* | par-each {7z a $"on/($in.name | path basename).zip" $in.name}
# same as above, but remove the root folder duplication
ls */* | par-each {cd $in.name; 7z a $"../../on/($in.name | path basename).zip" * ; cd -}
# grep replace
ls * | each {mv $in.name ($in.name | str replace dis diff)}