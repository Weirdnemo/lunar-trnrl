# Terrain data (not included)

The raw digital elevation models are large public products and are not stored in this
repository. Download them from the NASA Planetary Geodesy Data Archive (LOLA products):

- High-resolution LOLA topography for lunar south pole sites (5 m/pixel, `Site04`, `Site23`, `Haworth`):
  https://pgda.gsfc.nasa.gov/products/78
- 5 m/pixel LOLA polar DEM mosaics: https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/
- 250 m south polar DEM `HDEM_75S250M_ref2439p4km.tif`: LOLA gridded products at https://pgda.gsfc.nasa.gov/

Expected files in this folder:

    Site04_final_adj_5mpp_surf.tif
    Site23_final_adj_5mpp_surf.tif
    Haworth_final_adj_5mpp_surf.tif
    HDEM_75S250M_ref2439p4km.tif
    site04.npy, site23.npy        (crops; regenerate with scripts/crop_dem.py)
