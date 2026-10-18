"""Support-aware local contact morphology with explicit geometric descriptors.

Coordinates are zero-based half-open bins. Local boundaries summarize a selected
contact footprint and are not claims of exact biological domain boundaries.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, zoom, label

WIDTHS=(8,16,24,32,48,64,96)
FEATURE_NAMES=('centroid_axis','centroid_separation','principal_cos2','principal_sin2',
 'elongation','axis_spread','separation_spread','row_peak_fraction','column_peak_fraction',
 'anti_diagonal_peak_fraction','diagonal_peak_fraction','entropy','positive_area_fraction',
 'largest_component_fraction','component_count','edge_mass_fraction','interior_mass_fraction',
 'peak_to_mean','boundary_gradient','positive_negative_balance')

def normalized_field(matrix, expected, rows, bin_bp):
    matrix=np.asarray(matrix,float);n=len(matrix)
    d=abs(np.arange(n)[:,None]-np.arange(n)[None,:])
    rows=np.asarray(rows,bool)
    valid=(d*bin_bp>=200)&rows[:,None]&rows[None,:]
    oe=matrix/np.asarray(expected)[d].clip(1e-12)
    amplitude=float(oe[valid].mean()) if valid.any() else 0.
    field=np.where(valid,np.log1p(oe/max(amplitude,1e-12)),0.)
    def smooth(sigma):
        mass=gaussian_filter(valid.astype(float),sigma)
        return gaussian_filter(field,sigma)/mass.clip(1e-8)
    residual=smooth(1)-smooth(8)
    return field,residual,valid,oe

def integral(x):
    return np.pad(np.cumsum(np.cumsum(x,0),1),((1,0),(1,0)))

def box_sum(x,lo,hi):
    return x[hi,hi]-x[lo,hi]-x[hi,lo]+x[lo,lo]

def locate(residual,valid,minimum_distance_bins=2):
    """Search all fixed widths with a support-weighted positive ridge energy."""
    n=len(residual);tri=np.triu(valid,1)
    energy=integral(np.maximum(residual,0)*tri)
    count=integral(tri.astype(float));possible=integral(np.triu(np.ones((n,n)),minimum_distance_bins))
    best=None
    for width in WIDTHS:
        if width>n:continue
        starts=np.arange(0,n-width+1,4);ends=starts+width
        counts=box_sum(count,starts,ends);full=box_sum(possible,starts,ends)
        fraction=np.divide(counts,full,out=np.zeros_like(counts),where=full>0)
        scores=box_sum(energy,starts,ends)/np.maximum(counts,1)**.75
        scores[(fraction<.5)|(counts<12)]=-np.inf
        i=int(np.argmax(scores));proposal=(float(scores[i]),int(starts[i]),int(ends[i]),float(fraction[i]))
        if best is None or proposal[0]>best[0]:best=proposal
    if best is None or not np.isfinite(best[0]):return dict(lo=0,hi=n,score=0.,support_fraction=0.,valid=False)
    return dict(lo=best[1],hi=best[2],score=best[0],support_fraction=best[3],valid=True)

def describe(field,residual,valid,lo,hi):
    """Return dimensionless geometry and a mask-aware 32x32 aligned image."""
    z=residual[lo:hi,lo:hi];mask=np.triu(valid[lo:hi,lo:hi],1);n=len(z)
    weight=np.maximum(z,0)*mask
    mass=float(weight.sum());w=weight/max(mass,1e-12)
    y,x=np.indices(z.shape);xx=x/max(n-1,1);yy=y/max(n-1,1)
    axis=(xx+yy)/2;sep=(xx-yy)/np.sqrt(2)
    mx=float((w*xx).sum());my=float((w*yy).sum())
    cxx=float((w*(xx-mx)**2).sum());cyy=float((w*(yy-my)**2).sum());cxy=float((w*(xx-mx)*(yy-my)).sum())
    eigen=np.linalg.eigvalsh([[cxx,cxy],[cxy,cyy]]).clip(0)
    angle=.5*np.arctan2(2*cxy,cxx-cyy)
    row=w.sum(1);column=w.sum(0)
    anti=np.bincount((x+y).ravel(),weights=w.ravel(),minlength=2*n-1)
    diagonal=np.bincount((x-y+n).ravel(),weights=w.ravel(),minlength=2*n)
    threshold=float(np.quantile(weight[mask],.8)) if mask.any() else 0.
    binary=mask&(weight>max(threshold,1e-8));components,total=label(binary)
    sizes=np.bincount(components.ravel())[1:]
    edge=(xx<.125)|(xx>.875)|(yy<.125)|(yy>.875)
    grad=np.hypot(*np.gradient(field[lo:hi,lo:hi])) if n>=2 else np.zeros_like(z)
    vals=[float((w*axis).sum()),float((w*sep).sum()),float(np.cos(2*angle)),float(np.sin(2*angle)),
      float(np.sqrt((eigen[-1]+1e-8)/(eigen[0]+1e-8))),float(np.sqrt((w*(axis-(w*axis).sum())**2).sum())),
      float(np.sqrt((w*(sep-(w*sep).sum())**2).sum())),float(row.max()),float(column.max()),
      float(anti.max()),float(diagonal.max()),float(-(w[w>0]*np.log(w[w>0])).sum()/np.log(max(mask.sum(),2))),
      float(binary.sum()/max(mask.sum(),1)),float(sizes.max()/max(binary.sum(),1)) if len(sizes) else 0.,
      float(total)/max(n,1),float(w[edge].sum()),float(w[~edge].sum()),float(weight.max()/max(weight[mask].mean() if mask.any() else 0,1e-12)),
      float(grad[edge&mask].mean()) if (edge&mask).any() else 0.,float(mass/max(np.abs(z[mask]).sum(),1e-12))]
    crop=field[lo:hi,lo:hi];m=valid[lo:hi,lo:hi].astype(float)
    image=zoom(crop,(32/n,32/n),order=1);imask=zoom(m,(32/n,32/n),order=1)
    image=np.divide(image,imask,out=np.zeros_like(image),where=imask>.5)
    result=np.asarray(vals,float)
    if not np.isfinite(result).all() or not np.isfinite(image).all():raise ValueError('Nonfinite local geometry')
    return result,image.astype('float32')

def analyze(matrix,expected,rows,bin_bp):
    field,residual,valid,oe=normalized_field(matrix,expected,rows,bin_bp)
    region=locate(residual,valid,max(1,int(np.ceil(200/bin_bp))))
    geometry,image=describe(field,residual,valid,region['lo'],region['hi'])
    fixed_geometry,fixed_image=describe(field,residual,valid,0,len(field))
    width=region['hi']-region['lo']
    local_mask=valid[region['lo']:region['hi'],region['lo']:region['hi']]
    image_support=zoom(local_mask.astype(float),(32/width,32/width),order=1)>.9
    return dict(region=region,geometry=geometry,image=image,fixed_geometry=fixed_geometry,
                fixed_image=fixed_image,image_support=image_support,field=field,valid=valid,oe=oe)

def interval_iou(a,b):
    overlap=max(0,min(a[1],b[1])-max(a[0],b[0]));union=max(a[1],b[1])-min(a[0],b[0])
    return overlap/max(union,1)
