#!/bin/bash
#   Copyright (C) 2021 University of Oxford
#   SHCOPYRIGHT
#set -e
#set -x

######

if [ $# -lt 2 ] ; then
  echo "Usage: `basename $0` --FLAIR=<FLAIR_image_name> --T1=<T1_image_name> --outname=<output_basename> [--manualmask=<manualmask_name] [--nodistmaps] [--keepintermediate] [-v]"
  echo " "
  echo "The script applies the preprocessing pipeline on FLAIR, T1 and WM mask to be used in FSL truenet with a specified output basename"
  echo "FLAIR_image_name 		= 	absolute/relative path of the input unprocessed FLAIR image with the nifti file"
  echo "T1_image_name 			= 	absolute/relative path of the input unprocessed T1 image with the nifti file"
  echo "output_basename 		= 	absolute/relative path for the processed FLAIR and T1 images; output_basename_FLAIR.nii.gz, output_basename_T1.nii.gz etc. will be saved"
  echo "manualmask_name 		= 	absolute/relative path of the manual mask with the nifti file (only mandatory for creating masterfiles for training truenet)"
  echo "							Note: If FLAIR is provided, please provide manualmask in FLAIR space."
  echo "specify --nodistmaps if you want to skip adding distance maps in the masterfile (distmaps are only mandatory for creating masterfiles for training truenet)"
  echo "specify --keepintermediate if you want to save intermediate results"
  echo "-v for verbose"
  exit 0
fi

get_opt1() {
    arg=`echo $1 | sed 's/=.*//'`
    echo $arg
}

get_arg1() {
    if [ X`echo $1 | grep '='` = X ] ; then 
	echo "Option $1 requires an argument" 1>&2
	exit 1
    else 
	arg=`echo $1 | sed 's/.*=//'`
	if [ X$arg = X ] ; then
	    echo "Option $1 requires an argument" 1>&2
	    exit 1
	fi
	echo $arg
    fi
}

get_arg2() {
    if [ X$2 = X ] ; then
	echo "Option $1 requires an argument" 1>&2
	exit 1
    fi
    echo $2
}

# default options
argbase=
argmanualmask=
flairfile=
t1file=
verbose=no
keepint=no
distmaps=yes

if [ $# -eq 0 ] ; then usage; exit 0; fi
if [ $# -lt 2 ] ; then usage; exit 1; fi
niter=0;
while [ $# -ge 1 ] ; do
    niter=`echo $niter + 1 | bc`;
    iarg=`get_opt1 $1`;
    case "$iarg"
	in
	-v)
	    verbose=yes; 
	    shift;;
	--keepintermediate)
	    keepint=yes;
	    shift;;
	--nodistmaps)
	    distmaps=no;
	    shift;;
	--FLAIR)
	    flairfile=`get_arg1 $1`;
	    shift;;
	--T1)
	    t1file=`get_arg1 $1`;
	    shift;;
	--outname)
	    argbase=`get_arg1 $1`;
	    shift;;
	--manualmask)
	    argmanualmask=`get_arg1 $1`;
	    shift;;
	*)
	    usage;
	    echo "Unrecognised option $1" 1>&2
	    exit 1
    esac
done

if [ X${FSLDIR} = X ] ; then
    echo "ERROR: Cannot find FSL"
    echo "       Please setup the environment variable FSLDIR and try again"
    exit 1
fi

if [ X${flairfile} = X ] ; then
flairflag=0
else
flairflag=1
reqdheaders+=("FLAIR")      # Append 'FLAIR' header to the array
flairimg=`basename ${flairfile} .nii.gz`
flairdir=`dirname ${flairfile} `
pushd $flairdir > /dev/null
flairdir=`pwd`
popd > /dev/null
fi

if [ X${t1file} = X ] ; then
t1flag=0
else
t1flag=1
reqdheaders+=("T1")      # Append 'T1' header to the array
t1img=`basename ${t1file} .nii.gz`
t1dir=`dirname ${t1file} `
pushd $t1dir > /dev/null
t1dir=`pwd`
popd > /dev/null
fi

outname=`basename ${argbase}`
outdir=`dirname ${argbase}`
pushd $outdir > /dev/null
outdir=`pwd`
popd > /dev/null

if [ X${argmanualmask} = X ] ; then
manualflag=0
else
manualflag=1
reqdheaders+=("manualmask")      # Append 'manualmask' header to the array
manualmaskimg=`basename ${argmanualmask} .nii.gz`
manualmaskdir=`dirname ${argmanualmask} `
pushd ${manualmaskdir} > /dev/null
manualmaskdir=`pwd`
popd > /dev/null
fi

if [ ${distmaps} = yes ] ; then
distflag=1
reqdheaders+=("ventdistmap")      # Append 'ventdistmap' header to the array
reqdheaders+=("GMdistmap")      # Append 'GMdistmap' header to the array
else
distflag=0
fi

# SPECIFY ORIGINAL DIRECTORY
origdir=`pwd`

# CREATE TEMPORARY DIRECTORY
logID=`echo $(date | awk '{print $1 $2 $3}' |  sed 's/://g')`
TMPVISDIR=`mktemp -d ${outdir}/truenet_${logID}_${flairimg}_${t1img}_XXXXXX`

# BOTH FLAIR AND T1 IMAGES PROVIDED
######################################################################################
if [ ${flairflag} -eq 1 -a ${t1flag} -eq 1 ] ; then
# REORIENTING FLAIR AND T1 IMAGES TO STD SPACE
$FSLDIR/bin/fslreorient2std ${flairfile}.nii.gz ${TMPVISDIR}/FLAIR.nii.gz
$FSLDIR/bin/fslreorient2std ${t1file}.nii.gz ${TMPVISDIR}/T1.nii.gz
# PREPROCESSING OF FLAIR IMAGE
$FSLDIR/bin/bet ${TMPVISDIR}/FLAIR.nii.gz ${TMPVISDIR}/FLAIR_brain.nii.gz
$FSLDIR/bin/fast -B --nopve ${TMPVISDIR}/FLAIR_brain.nii.gz 
${FSLDIR}/bin/imcp ${TMPVISDIR}/FLAIR_brain_restore.nii.gz ${outdir}/${outname}_FLAIR.nii.gz
# APPLYING FSL_ANAT ON THE REORIENTED T1 IMAGE
$FSLDIR/bin/fsl_anat --nosubcortseg -i ${TMPVISDIR}/T1.nii.gz
# REGISTERING T1 TO FLAIR IMAGE
$FSLDIR/bin/flirt -dof 6 -in ${TMPVISDIR}/T1.anat/T1_biascorr_brain.nii.gz -ref ${TMPVISDIR}/FLAIR_brain.nii.gz -out ${TMPVISDIR}/${outname}_T1_brain_2FLAIR.nii.gz
$FSLDIR/bin/flirt -dof 6 -in ${TMPVISDIR}/T1.anat/T1_biascorr_brain.nii.gz -ref ${TMPVISDIR}/FLAIR_brain.nii.gz -omat ${TMPVISDIR}/${outname}_T1_brain_2FLAIR.mat
$FSLDIR/bin/imcp ${TMPVISDIR}/${outname}_T1_brain_2FLAIR.nii.gz ${outdir}/${outname}_T1.nii.gz
# GETTING WM MASK FOR POSTPROCESSING
$FSLDIR/bin/make_bianca_mask ${TMPVISDIR}/T1.anat/T1_biascorr ${TMPVISDIR}/T1.anat/T1_fast_pve_0.nii.gz ${TMPVISDIR}/T1.anat/MNI_to_T1_nonlin_field.nii.gz 1
$FSLDIR/bin/flirt -dof 6 -in ${TMPVISDIR}/T1.anat/T1_biascorr_bianca_mask.nii.gz -ref ${TMPVISDIR}/FLAIR_brain.nii.gz -applyxfm -init ${TMPVISDIR}/${outname}_T1_brain_2FLAIR.mat -out ${TMPVISDIR}/${outname}_WMmask_2FLAIR.nii.gz
$FSLDIR/bin/fslmaths ${TMPVISDIR}/${outname}_WMmask_2FLAIR.nii.gz -thr 0.5 -bin ${outdir}/${outname}_WMmask.nii.gz
# BINARISING MANUALMASK (just to binarize and copy to the required directory)
if [ ${manualflag} -eq 1 ] ; then
$FSLDIR/bin/fslmaths ${argmanualmask} -thr -0.1 -bin ${outdir}/${outname}_manualmask.nii.gz
fi

# GETTING DISTANCE MAPS IF REQUIRED
if [ ${distflag} -eq 1 ] ; then
$FSLDIR/bin/flirt -dof 6 -in ${TMPVISDIR}/T1.anat/T1_biascorr_ventmask.nii.gz -ref ${TMPVISDIR}/FLAIR_brain.nii.gz -applyxfm -init ${TMPVISDIR}/${outname}_T1_brain_2FLAIR.mat -out ${TMPVISDIR}/${outname}_ventmask_2FLAIR.nii.gz
$FSLDIR/bin/fslmaths ${TMPVISDIR}/${outname}_ventmask_2FLAIR.nii.gz -thr 0.2 -bin ${TMPVISDIR}/${outname}_ventmask_2FLAIR_bin.nii.gz
$FSLDIR/bin/fslmaths ${outdir}/${outname}_T1.nii.gz -bin ${outdir}/${outname}_brainmask.nii.gz
# -m option to be deprecated soon in the next release of FSL: to be re-evaulated later.
$FSLDIR/bin/distancemap -i ${TMPVISDIR}/${outname}_ventmask_2FLAIR_bin.nii.gz -o ${TMPVISDIR}/${outname}_T1_2FLAIR_ventdistmap_full.nii.gz
$FSLDIR/bin/fslmaths ${TMPVISDIR}/${outname}_T1_2FLAIR_ventdistmap_full.nii.gz -mas ${outdir}/${outname}_brainmask.nii.gz ${TMPVISDIR}/${outname}_T1_2FLAIR_ventdistmap.nii.gz
$FSLDIR/bin/fslmaths ${TMPVISDIR}/${outname}_T1_2FLAIR_ventdistmap.nii.gz -thr -1 -uthr 6 -bin -fillh26 ${TMPVISDIR}/${outname}_extended_ventricles_2FLAIR.nii.gz
$FSLDIR/bin/fslmaths ${outdir}/${outname}_WMmask.nii.gz -add ${TMPVISDIR}/${outname}_extended_ventricles_2FLAIR.nii.gz -thr 0 -bin ${TMPVISDIR}/${outname}_nonGMmask_2FLAIR.nii.gz
$FSLDIR/bin/fslmaths ${outdir}/${outname}_brainmask.nii.gz -sub ${TMPVISDIR}/${outname}_nonGMmask_2FLAIR.nii.gz -thr 0 -bin ${TMPVISDIR}/${outname}_GMmask_2FLAIR.nii.gz
# -m option to be deprecated soon in the next release of FSL: to be re-evaulated later.
$FSLDIR/bin/distancemap -i ${TMPVISDIR}/${outname}_GMmask_2FLAIR.nii.gz -o ${TMPVISDIR}/${outname}_T1_2FLAIR_GMdistmap_full.nii.gz
$FSLDIR/bin/fslmaths ${TMPVISDIR}/${outname}_T1_2FLAIR_GMdistmap_full.nii.gz -mas ${outdir}/${outname}_brainmask.nii.gz ${TMPVISDIR}/${outname}_T1_2FLAIR_GMdistmap.nii.gz
$FSLDIR/bin/fslmaths ${TMPVISDIR}/${outname}_T1_2FLAIR_ventdistmap.nii.gz -mas ${outdir}/${outname}_WMmask.nii.gz ${outdir}/${outname}_ventdistmap.nii.gz
$FSLDIR/bin/fslmaths ${TMPVISDIR}/${outname}_T1_2FLAIR_GMdistmap.nii.gz -mas ${outdir}/${outname}_WMmask.nii.gz ${outdir}/${outname}_GMdistmap.nii.gz
fi

if [ ${keepint} = yes ] ; then
# SAVING INTERMEDIATE FILES
######################################################################################
mv ${TMPVISDIR} ${outdir}/${outname}_temp/
fi
# ONLY FLAIR IMAGES PROVIDED
######################################################################################
elif [ ${flairflag} -eq 1 -a ${t1flag} -eq 0 ] ; then
# REORIENTING FLAIR AND T1 IMAGES TO STD SPACE
$FSLDIR/bin/fslreorient2std ${flairfile}.nii.gz ${TMPVISDIR}/${outname}_FLAIR.nii.gz
# PREPROCESSING OF FLAIR IMAGE
$FSLDIR/bin/bet ${TMPVISDIR}/${outname}_FLAIR.nii.gz ${TMPVISDIR}/${outname}_FLAIR_brain.nii.gz -m
$FSLDIR/bin/fast -n 2 -B ${TMPVISDIR}/${outname}_FLAIR_brain.nii.gz 
${FSLDIR}/bin/imcp ${TMPVISDIR}/${outname}_FLAIR_brain_restore.nii.gz ${outdir}/${outname}_FLAIR.nii.gz
# REGISTERING FLAIR IMAGE TO MNI SPACE
$FSLDIR/bin/flirt -in ${FSLDIR}/data/standard/MNI152_T1_2mm_brain.nii.gz -ref ${outdir}/${outname}_FLAIR.nii.gz -omat ${TMPVISDIR}/${outname}_MNI_2FLAIR.mat
# GETTING WM MASK FOR POSTPROCESSING
make_WMmask_flair ${TMPVISDIR}/${outname}_FLAIR.nii.gz ${TMPVISDIR}/${outname}_FLAIR_brain_pve_0.nii.gz ${TMPVISDIR}/${outname}_MNI_2FLAIR.mat 1
$FSLDIR/bin/fslmaths ${TMPVISDIR}/${outname}_FLAIR_WMmask.nii.gz -thr 0.5 -bin ${outdir}/${outname}_WMmask.nii.gz
# BINARISING MANUALMASK (just to binarize and copy to the required directory)
if [ ${manualflag} -eq 1 ] ; then
$FSLDIR/bin/fslmaths ${argmanualmask} -thr -0.1 -bin ${outdir}/${outname}_manualmask.nii.gz
fi
# GETTING DISTANCE MAPS IF REQUIRED
if [ ${distflag} -eq 1 ] ; then
$FSLDIR/bin/fslmaths ${TMPVISDIR}/${outname}_FLAIR_ventmask.nii.gz -thr 0.2 -bin ${TMPVISDIR}/${outname}_FLAIR_ventmask_bin.nii.gz
$FSLDIR/bin/imcp ${TMPVISDIR}/${outname}_FLAIR_brain_mask.nii.gz ${outdir}/${outname}_brainmask.nii.gz
# -m option to be deprecated soon in the next release of FSL: to be re-evaulated later.
$FSLDIR/bin/distancemap -i ${TMPVISDIR}/${outname}_FLAIR_ventmask_bin.nii.gz -o ${outdir}/${outname}_ventdistmap.nii.gz
$FSLDIR/bin/fslmaths ${outdir}/${outname}_ventdistmap.nii.gz -mas ${outdir}/${outname}_brainmask.nii.gz ${outdir}/${outname}_ventdistmap.nii.gz
$FSLDIR/bin/fslmaths ${outdir}/${outname}_ventdistmap.nii.gz -thr -1 -uthr 3 -bin -fillh26 ${TMPVISDIR}/${outname}_FLAIR_extended_ventricles.nii.gz
$FSLDIR/bin/fslmaths ${outdir}/${outname}_WMmask.nii.gz -add ${TMPVISDIR}/${outname}_FLAIR_extended_ventricles.nii.gz -thr 0 -bin ${TMPVISDIR}/${outname}_FLAIR_nonGMmask.nii.gz
$FSLDIR/bin/fslmaths ${outdir}/${outname}_brainmask.nii.gz -sub ${TMPVISDIR}/${outname}_FLAIR_nonGMmask.nii.gz -thr 0 -bin ${TMPVISDIR}/${outname}_FLAIR_GMmask.nii.gz
# -m option to be deprecated soon in the next release of FSL: to be re-evaulated later.
$FSLDIR/bin/distancemap -i ${TMPVISDIR}/${outname}_FLAIR_GMmask.nii.gz -o ${outdir}/${outname}_GMdistmap.nii.gz
$FSLDIR/bin/fslmaths ${outdir}/${outname}_GMdistmap.nii.gz -mas ${outdir}/${outname}_brainmask.nii.gz ${outdir}/${outname}_GMdistmap.nii.gz
$FSLDIR/bin/fslmaths ${outdir}/${outname}_ventdistmap.nii.gz -mas ${outdir}/${outname}_WMmask.nii.gz ${outdir}/${outname}_ventdistmap.nii.gz
$FSLDIR/bin/fslmaths ${outdir}/${outname}_GMdistmap.nii.gz -mas ${outdir}/${outname}_WMmask.nii.gz ${outdir}/${outname}_GMdistmap.nii.gz
fi

if [ ${keepint} = yes ] ; then
# SAVING INTERMEDIATE FILES
######################################################################################
mv ${TMPVISDIR}/ ${outdir}/${outname}_temp/
fi
# ONLY T1 IMAGES PROVIDED
######################################################################################
elif [ ${t1flag} -eq 1 -a ${flairflag} -eq 0 ] ; then
# REORIENTING FLAIR AND T1 IMAGES TO STD SPACE
$FSLDIR/bin/fslreorient2std ${t1file}.nii.gz ${TMPVISDIR}/T1.nii.gz
# APPLYING FSL_ANAT ON THE REORIENTED T1 IMAGE
$FSLDIR/bin/fsl_anat --nosubcortseg -i ${TMPVISDIR}/T1.nii.gz
$FSLDIR/bin/imcp ${TMPVISDIR}/T1.anat/T1_biascorr_brain.nii.gz ${outdir}/${outname}_T1.nii.gz
# BINARISING MANUALMASK (just to binarize and copy to the required directory)
if [ ${manualflag} -eq 1 ] ; then
$FSLDIR/bin/fslmaths ${argmanualmask} -thr -0.1 -bin ${outdir}/${outname}_manualmask.nii.gz
fi
# GETTING WM MASK FOR POSTPROCESSING
$FSLDIR/bin/make_bianca_mask ${TMPVISDIR}/T1.anat/T1_biascorr ${TMPVISDIR}/T1.anat/T1_fast_pve_0.nii.gz ${TMPVISDIR}/T1.anat/MNI_to_T1_nonlin_field.nii.gz 1
$FSLDIR/bin/imcp ${TMPVISDIR}/T1.anat/T1_biascorr_bianca_mask.nii.gz ${outdir}/${outname}_WMmask.nii.gz 
# GETTING DISTANCE MAPS IF REQUIRED
if [ ${distflag} -eq 1 ] ; then
$FSLDIR/bin/fslmaths ${TMPVISDIR}/T1.anat/T1_biascorr_ventmask.nii.gz -bin ${TMPVISDIR}/${outname}_T1_ventmask_bin.nii.gz
$FSLDIR/bin/fslmaths ${outdir}/${outname}_T1.nii.gz -bin ${outdir}/${outname}_brainmask.nii.gz
$FSLDIR/bin/distancemap -i ${TMPVISDIR}/${outname}_T1_ventmask_bin.nii.gz -o ${TMPVISDIR}/${outname}_T1_ventdistmap.nii.gz
$FSLDIR/bin/fslmaths ${TMPVISDIR}/${outname}_T1_ventdistmap.nii.gz -mas ${outdir}/${outname}_brainmask.nii.gz ${TMPVISDIR}/${outname}_T1_ventdistmap.nii.gz
$FSLDIR/bin/fslmaths ${TMPVISDIR}/${outname}_T1_ventdistmap.nii.gz -thr -1 -uthr 3 -bin -fillh26 ${TMPVISDIR}/${outname}_T1_extended_ventricles.nii.gz
$FSLDIR/bin/fslmaths ${outdir}/${outname}_WMmask.nii.gz -add ${TMPVISDIR}/${outname}_T1_extended_ventricles.nii.gz -thr 0 -bin ${TMPVISDIR}/${outname}_T1_nonGMmask.nii.gz
$FSLDIR/bin/fslmaths ${outdir}/${outname}_brainmask.nii.gz -sub ${TMPVISDIR}/${outname}_T1_nonGMmask.nii.gz -thr 0 -bin ${TMPVISDIR}/${outname}_T1_GMmask.nii.gz
$FSLDIR/bin/distancemap -i ${TMPVISDIR}/${outname}_T1_GMmask.nii.gz -o ${TMPVISDIR}/${outname}_T1_GMdistmap.nii.gz
$FSLDIR/bin/fslmaths ${TMPVISDIR}/${outname}_T1_GMdistmap.nii.gz -mas ${outdir}/${outname}_brainmask.nii.gz ${TMPVISDIR}/${outname}_T1_GMdistmap.nii.gz
$FSLDIR/bin/fslmaths ${TMPVISDIR}/${outname}_T1_ventdistmap.nii.gz -mas ${outdir}/${outname}_WMmask.nii.gz ${outdir}/${outname}_ventdistmap.nii.gz
$FSLDIR/bin/fslmaths ${TMPVISDIR}/${outname}_T1_GMdistmap.nii.gz -mas ${outdir}/${outname}_WMmask.nii.gz ${outdir}/${outname}_GMdistmap.nii.gz
fi

if [ ${keepint} = yes ] ; then
# SAVING INTERMEDIATE FILES
######################################################################################
mv ${TMPVISDIR}/ ${outdir}/${outname}_temp/
fi
fi

# REMOVES TEMPORARY DIRECTORY 
if [ ${keepint} != yes ] ; then
rm -r ${TMPVISDIR}
fi
exit 0














