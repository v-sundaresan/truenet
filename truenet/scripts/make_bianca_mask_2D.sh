#!/bin/bash
#set -e
#set -x

function cluster2D {
    # Usage: cluster2D <input> <output> <axis> <threshold>
    #  e.g. cluster2D invol outvol y 0.9
 invol=$1
 outvol=$2
 axis=$3
 thresh=$4
 tmpdir=`$FSLDIR/bin/tmpnam`
 rm $tmpdir
 mkdir $tmpdir
 $FSLDIR/bin/fslsplit $invol $tmpdir/cl2d -$axis
 for fn in $tmpdir/cl2d* ; do
     ff=`$FSLDIR/bin/remove_ext $fn`
     $FSLDIR/bin/cluster -i $fn -t $thresh --connectivity=6 --no_table --oindex=${ff}_IDX
 done
 $FSLDIR/bin/fslmerge -$axis $outvol $tmpdir/*_IDX*
 rm -f $tmpdir/cl2d*
 rmdir $tmpdir
}


function select_clusters2D {
    # Usage: select_clusters2D <partial index mask> <cluster index image> <axis> <selected cluster image, as mask>
    #  e.g. select_clusters2D partial_vent_mask CSFseg_pve_0_idx y vent_mask
 partmask=$1
 idx=$2
 axis=$3
 outvol=$4
 sizethresh=10   # in voxels
 tmpdir=`$FSLDIR/bin/tmpnam`
 rm $tmpdir
 mkdir $tmpdir
 $FSLDIR/bin/fslsplit $idx $tmpdir/scl2d -$axis
 $FSLDIR/bin/fslsplit $partmask $tmpdir/mask2d -$axis
 for fn in $tmpdir/scl2d* ; do
     ff=`$FSLDIR/bin/remove_ext $fn`
     nn=`basename $ff | sed 's/scl2d//'`
     $FSLDIR/bin/fslmaths $fn -mas $tmpdir/mask2d${nn} ${ff}_idxmasked
     # get indices and sizes of two biggest clusters with some overlap
     touch $tmpdir/sortlist.txt
     rm $tmpdir/sortlist.txt
     touch $tmpdir/sortlist.txt
     idx1=`$FSLDIR/bin/fslstats ${ff}_idxmasked -P 100`
     idx0=`echo $idx1 + 1 | bc`
     # step down through the non-zero indices
     while [ `echo "$idx1 > 0" | bc -l` -eq 1 ] ; do
	 idx2=`$FSLDIR/bin/fslstats ${ff}_idxmasked -u $idx1 -P 100`
	 n=`$FSLDIR/bin/fslstats ${ff}_idxmasked -l $idx2 -u $idx0 -V | awk '{ print $1 }'`
	 if [ $n -ge $sizethresh ] ; then
	     echo "$n $idx1" >> $tmpdir/sortlist.txt
	 fi
	 idx0=$idx1
	 idx1=$idx2
     done
     # take top two results
     idxlist=`sort -n $tmpdir/sortlist.txt | tail -2 | awk '{ print $2 }' | sort -u` 
     $FSLDIR/bin/fslmaths $fn -mul 0 ${ff}_SEL
     for idx in $idxlist ; do
 	 $FSLDIR/bin/fslmaths ${fn} -thr $idx -uthr $idx -bin -add ${ff}_SEL ${ff}_SEL
     done
 done
 $FSLDIR/bin/fslmerge -$axis $outvol $tmpdir/*_SEL*
 rm -f $tmpdir/scl2d* $tmpdir/mask2d* $tmpdir/sortlist.txt
 rmdir $tmpdir
}

function distancemap2D {
 # Usage: distancemap2D <in> <mask> <axis> <out>
 invol=$1
 maskvol=$2
 axis=$3
 outvol=$4
 bigdist=100
 tmpdir=`$FSLDIR/bin/tmpnam`
 rm $tmpdir
 mkdir $tmpdir
 $FSLDIR/bin/fslsplit $invol $tmpdir/in2d -$axis
 $FSLDIR/bin/fslsplit $maskvol $tmpdir/mask2d -$axis
 for fn in $tmpdir/in2d* ; do
     ff=`$FSLDIR/bin/remove_ext $fn`
     nn=`basename $ff | sed 's/in2d//'`
     maxv=`$FSLDIR/bin/fslstats $fn -P 100`
     if [ `echo "$maxv > 0" | bc -l` -eq 1 ] ; then
	 $FSLDIR/bin/distancemap -i $fn -m $tmpdir/mask2d${nn} -o ${ff}_dist
     else
	 $FSLDIR/bin/fslmaths $fn -mul 0 -add $bigdist ${ff}_dist
     fi
 done
 $FSLDIR/bin/fslmerge -$axis $outvol $tmpdir/*_dist*
 rm -f $tmpdir/in2d* $tmpdir/mask2d*
 rmdir $tmpdir
  
}


function distance2Dlabel {
 # Usage: distancemap2D <label> <mask> <axis> <out>
 labvol=$1
 maskvol=$2
 axis=$3
 outvol=$4
 bigdist=100
 tmpdir=`$FSLDIR/bin/tmpnam`
 rm $tmpdir
 mkdir $tmpdir
 $FSLDIR/bin/fslsplit $labvol $tmpdir/lab2d -$axis
 $FSLDIR/bin/fslsplit $maskvol $tmpdir/mask2d -$axis
 for fn in $tmpdir/lab2d* ; do
     ff=`$FSLDIR/bin/remove_ext $fn`
     nn=`basename $ff | sed 's/lab2d//'`
     maxv=`$FSLDIR/bin/fslstats $fn -P 100`
     if [ `echo "$maxv > 0" | bc -l` -eq 1 ] ; then
	 $FSLDIR/bin/distancemap -i $fn -m $tmpdir/mask2d${nn} --interp=$fn -o ${ff}_dist
     else
	 $FSLDIR/bin/fslmaths $fn -mul 0 -add $bigdist ${ff}_dist
     fi
 done
 $FSLDIR/bin/fslmerge -$axis $outvol $tmpdir/*_dist*
 rm -f $tmpdir/lab2d* $tmpdir/mask2d*
 rmdir $tmpdir
  
}

################################################################################################################


if [ $# -lt 1 ] ; then
  echo "Usage: `basename $0` <brain extracted image> <CSF pve image> <image to standard MAT>"
  echo "       e.g. $0 FLAIR_brain FLAIR_brain_pve_0 FLAIR2STD.mat "
  echo " "
  echo "This script creates files called GMmaskexcl.nii.gz and vent_mask.nii.gz"
  echo "(Additional outputs: inverted transform, brain mask)"
  echo "The GMmaskexcl.nii.gz mask can be used to remove candidate lesions from the automatic segmentation via fslmaths:"
  echo "  e.g.  fslmaths bianca_output -mas GMmaskexcl bianca_masked"
  echo "Requires mask files GMWMmask.nii.gz and HOlatvent.nii.gz to be in the same folder of the script"
  exit 0
fi

IMAGE=`$FSLDIR/bin/remove_ext $1`
CSFPVE=`$FSLDIR/bin/remove_ext $2`
#MAT2STD=`$FSLDIR/bin/remove_ext $3`
MAT2STD=$(echo "$3" | cut -f 1 -d '.')

# Dependencies: in the same folder of the script I need GMWMmask and HOlatvent
GMWMmask=`dirname $0`/GMWMmask.nii.gz
HOlatvent=`dirname $0`/HOlatvent.nii.gz

$FSLDIR/bin/fslmaths ${IMAGE} -bin ${IMAGE}_mask

# Transform masks from standard space to subject's space
$FSLDIR/bin/convert_xfm -omat ${MAT2STD}_inv.mat -inverse ${MAT2STD}.mat
$FSLDIR/bin/flirt -in $HOlatvent -ref ${IMAGE} -out HOlatvent2${IMAGE} -applyxfm -init ${MAT2STD}_inv.mat -interp nearestneighbour
$FSLDIR/bin/flirt -in $GMWMmask -ref ${IMAGE} -out subcortexcl2${IMAGE} -applyxfm -init ${MAT2STD}_inv.mat
$FSLDIR/bin/fslmaths subcortexcl2${IMAGE} -thr 0.5 -bin subcortexcl2${IMAGE}

# Clean CSF mask to remove subcortical parts and bright FLAIR voxels
#   separate current CSF mask to exclude subcortical structures 
$FSLDIR/bin/fslmaths ${CSFPVE} -mas subcortexcl2${IMAGE} ${CSFPVE}_masked
#   eliminate bright FLAIR (the second mode for the widest distribution - a known FAST problem)
$FSLDIR/bin/fslmaths ${CSFPVE}_masked -thr 0.9 -binv -mul ${IMAGE} ${IMAGE}_nonCSF09
medval=`$FSLDIR/bin/fslstats ${IMAGE}_nonCSF09 -P 50`
$FSLDIR/bin/fslmaths ${IMAGE} -uthr $medval -bin -mul ${CSFPVE}_masked ${CSFPVE}_masked

# Remove ventricles from this mask
#   threshold CSF pve at 0.9 and cluster in each slice (coronal)
cluster2D ${CSFPVE}_masked ${CSFPVE}_masked_idx y 0.9
# detect clusters that overlap "significantly" (percentage of mask _and_ size threshold?) with the HO ventricle mask and classify these as ventricular parts and split the 0.9 CSF into ventricular and non-ventricular parts
$FSLDIR/bin/fslmaths ${CSFPVE}_masked_idx -mas HOlatvent2${IMAGE} ${CSFPVE}_masked_idx_masked
select_clusters2D ${CSFPVE}_masked_idx_masked ${CSFPVE}_masked_idx y vent_mask

# # # # # # # # improved ventricles segmentation
# As normally the posterior portions of the ventricles are those that are not identified after nonlinear registration as they're the most distorted,
# the idea is to start close to (3 slices before) the most posterior CORONAL slice where ventricles where identified and move towards the occipital lobe, adding,
# slice by slice, the 2 biggest clusters (bigger than 5 voxels) of the thresholded (0.9) csf pve (pve0) that overlap with the ventricles identified in the previous slice.
if [ ! -d tmpdir ]; then
mkdir tmpdir
fi 
imcp vent_mask tmpdir/ventmask_temp

nlast_AP=`fslstats tmpdir/ventmask_temp -w | awk '{ print $3 }' `
nlastAP=`echo $nlast_AP + 3 | bc`
$FSLDIR/bin/fslroi tmpdir/ventmask_temp tmpdir/ventmask_lastCORslice 0 -1 ${nlastAP} 1 0 -1
VENTlastcorslice=tmpdir/ventmask_lastCORslice
# cut the vent mask until that slice (one slice earlier). From that one on, I’ll add the coronal slices with additional portions of ventricles
totdim_AP=`fslval tmpdir/ventmask_temp dim2`
startslice=`echo $nlast_AP + 4 | bc`
size_ventmaskOK=`echo $totdim_AP - $startslice | bc`
$FSLDIR/bin/fslroi tmpdir/ventmask_temp tmpdir/ventmask_anterior 0 -1 $startslice $size_ventmaskOK 0 -1
#from startslice towards 0 (posterior) I want to look for the ventricles in the pve0 image thresholded at 0.9
$FSLDIR/bin/fslmaths ${CSFPVE}_masked -thr 0.9  tmpdir/brain_pve_0_thr09
# step down until the first (zero) slice
idslice1=$nlastAP
stop=0
while [ X`echo "if ( 0 <= $idslice1 ) { 1 }" | bc -l` = X1 ] && [ $stop = 0 ] ; do 
    #echo $idslice1
    $FSLDIR/bin/fslroi tmpdir/brain_pve_0_thr09  tmpdir/pve09_lastCORslice 0 -1 $idslice1 1 0 -1    
    $FSLDIR/bin/cluster -i tmpdir/pve09_lastCORslice -t 0.05 -o tmpdir/pve09_lastCORslice_idx --connectivity=6 --no_table
    $FSLDIR/bin/fslcpgeom tmpdir/pve09_lastCORslice $VENTlastcorslice
    $FSLDIR/bin/fslmaths tmpdir/pve09_lastCORslice_idx -mas $VENTlastcorslice tmpdir/pve09_lastCORslice_idx_masked

    idx1=`$FSLDIR/bin/fslstats tmpdir/pve09_lastCORslice_idx_masked -P 100 | bc`
    # if the slice is empty, there are no more ventricles to add, so it will fill the rest of the image with zero slices
    if [ $idx1 = 0 ] ; then
	pad=`echo $idslice1 +1 | bc`
	#echo no more ventricles - padding with zeros: fslroi tmpdir/ventmask_temp tmpdir/ventmask_posterior 0 -1 0 $pad 0 -1
	$FSLDIR/bin/fslroi tmpdir/ventmask_temp tmpdir/ventmask_posterior 0 -1 0 $pad 0 -1
	$FSLDIR/bin/fslmerge -y tmpdir/ventmask_anterior tmpdir/ventmask_posterior tmpdir/ventmask_anterior
	stop=1
    else
    # get indices and sizes of two biggest clusters with some overlap
    touch tmpdir/sortlist.txt
    rm tmpdir/sortlist.txt
    touch tmpdir/sortlist.txt
    idx0=`echo $idx1 + 1 | bc`
    # step down through the non-zero indices
    while [ X`echo "if ( 0 < $idx1 ) { 1 }" | bc -l` = X1 ] ; do
	idx2=`$FSLDIR/bin/fslstats tmpdir/pve09_lastCORslice_idx_masked -u $idx1 -P 100`
	n=`$FSLDIR/bin/fslstats tmpdir/pve09_lastCORslice_idx_masked -l $idx2 -u $idx0 -V | awk '{ print $1 }'`
	if [ $n -ge 5 ] ; then
	    echo "$n $idx1" >> tmpdir/sortlist.txt
	fi
	idx0=$idx1
	idx1=$idx2
    done
    # take top two results
    idxlist=`sort -n tmpdir/sortlist.txt | tail -2 | awk '{ print $2 }' | sort -u`
    $FSLDIR/bin/fslmaths tmpdir/pve09_lastCORslice_idx -mul 0 tmpdir/SEL
    for idx in $idxlist ; do
	$FSLDIR/bin/fslmaths tmpdir/pve09_lastCORslice_idx -thr $idx -uthr $idx -bin -add tmpdir/SEL tmpdir/SEL
    done
    $FSLDIR/bin/fslmerge -y tmpdir/ventmask_anterior tmpdir/SEL tmpdir/ventmask_anterior
    VENTlastcorslice=tmpdir/SEL    
    idslice1=`echo $idslice1 -1 | bc`
    fi
done
mv tmpdir/ventmask_anterior.nii.gz tmpdir/ventmask_unfilled.nii.gz
# fill holes in the ventricle mask (try to include calcifications in the ventricles)
$FSLDIR/bin/fslmaths tmpdir/ventmask_unfilled.nii.gz -fillh26 tmpdir/ventmask.nii.gz
# # # # # # #

imcp tmpdir/ventmask vent_mask

$FSLDIR/bin/fslmaths ${CSFPVE}_masked_idx -bin -sub vent_mask nonvent_mask
# take the 0.1 CSF and calculate distance map to both ventricular and non-ventricular parts
$FSLDIR/bin/fslmaths ${CSFPVE}_masked -thr 0.1 -bin ${CSFPVE}_masked_thr01
# add outer rim to nonvent mask for calculating distances

$FSLDIR/bin/fslmaths ${IMAGE}_mask -dilF -sub ${IMAGE}_mask ${IMAGE}_mask_outerrim
$FSLDIR/bin/fslmaths nonvent_mask -add ${IMAGE}_mask_outerrim -bin nonvent_mask
distancemap2D vent_mask ${CSFPVE}_masked_thr01 y dist2vent
distancemap2D nonvent_mask ${CSFPVE}_masked_thr01 y dist2nonvent
# remove all voxels (not clusters, but voxels) where distance to ventricles is less than to non-ventricular parts (need to see how bad this is at removing sulcal CSF voxels)
$FSLDIR/bin/fslmaths dist2vent -sub dist2nonvent -bin -mul ${CSFPVE}_masked_thr01 CSFmask


# Add an outerrim from the brain mask and generate use the single biggest (3D) cluster as the basis of the cortical CSF
$FSLDIR/bin/fslmaths CSFmask -mas ${IMAGE}_mask -add ${IMAGE}_mask_outerrim CSFmask
# Find single biggest 3D cluster and the dilate this by 4mm (in 2D due to massive slice thickness)
$FSLDIR/bin/cluster -i CSFmask.nii.gz -t 0.5 --oindex=CSF_index --connectivity=6 --no_table
$FSLDIR/bin/fslroi CSFmask.nii.gz kernelGMcortex2D 0 21 0 1 0 21 0 1
$FSLDIR/bin/fslmaths kernelGMcortex2D.nii.gz -mul 0 -add 1 -roi 10 1 0 1 10 1 0 1 -kernel sphere 4 -dilF kernelGMcortex2D.nii.gz 
$FSLDIR/bin/fslmaths CSF_index -thr `$FSLDIR/bin/fslstats CSF_index -P 100` -bin CSFmask1
$FSLDIR/bin/fslmaths CSFmask1 -kernel file kernelGMcortex2D.nii.gz -dilF -binv GMmaskexcl

#remove background 1 and ventricles
$FSLDIR/bin/fslmaths GMmaskexcl.nii.gz -mul ${IMAGE}_mask -sub vent_mask.nii.gz -thr 0.5 -bin GMmaskexcl

rm -r tmpdir
rm ${IMAGE}_mask_outerrim.nii.gz nonvent_mask.nii.gz dist2nonvent.nii.gz dist2vent.nii.gz kernelGMcortex2D.nii.gz CSFmask.nii.gz CSFmask1.nii.gz CSF_index.nii.gz
rm HOlatvent2${IMAGE}.nii.gz  subcortexcl2${IMAGE}.nii.gz ${IMAGE}_nonCSF09.nii.gz ${CSFPVE}_masked.nii.gz ${CSFPVE}_masked_idx.nii.gz ${CSFPVE}_masked_idx_masked.nii.gz ${CSFPVE}_masked_thr01.nii.gz
echo Done
exit 0