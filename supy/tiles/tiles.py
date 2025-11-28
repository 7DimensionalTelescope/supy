
import os
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table, Column
from datetime import datetime
from astropy.table import vstack
from typing import List, Union
from .coord import parse_coordinates
from ..const import REFDATA_DIR

class Tiles:
    def __init__(self, tile_path: str = None):
        if tile_path is None:
            self.tile_path = os.path.join(REFDATA_DIR, 'tileinfo/7-DT/final_tiles.txt')
        else:
            self.tile_path = tile_path
        self.tbl_RIS = None
        self.coords_RIS = None

    @property
    def target_table(self):
        if not(hasattr(self, '_target_table')):
            print("Target table not produced. Run find_overlapping_tiles first.")
            return None
        return self._target_table
    
    @property
    def tile_table(self):
        if not(hasattr(self, '_tile_table')):
            print("Tile table not produced. Run find_overlapping_tiles first.")
            return None
        return self._tile_table
    
    def find_overlapping_tiles(self, 
                            loc=None,
                            j2000=None,
                            ra=None, 
                            dec=None, 
                            aperture= 0,
                            match_tolerance_minutes = 4,
                            fraction_overlap_lower = 0.2,
                            visualize=False,
                            **kwargs):
        """
        Find the tiles that overlap with the given coordinates and aperture sizes.

        Parameters:
        - list_ra: list of RA coordinates
        - list_dec: list of Dec coordinates
        - list_aperture: list of aperture sizes in degrees or single value (default is 0 for point matching)
        - visualize (bool): Whether to visualize the overlapping tiles
        - visualize_ncols (int): Number of columns in the visualization grid

        Returns:
        - A table containing the overlapping or innermost tiles for each coordinate.
        """
        if loc is not None:
            try:
                loc = np.atleast_2d(loc)
                ra, dec = loc.T
            except:
                coords = [parse_coordinates(l) for l in loc]
                ra, dec = zip(*coords)

        if j2000 is not None:
            list_j2000 = np.atleast_1d(j2000)
            coords = [parse_coordinates(j2000) for j2000 in list_j2000]
            ra, dec = zip(*coords)

        list_ra = np.atleast_1d(ra)
        list_dec = np.atleast_1d(dec)
        
        # Ensure list_aperture matches list_ra and list_dec
        if isinstance(aperture, (int, float)):
            list_aperture = [aperture] * len(list_ra)
        elif len(aperture) != len(list_ra):
            raise ValueError("list_aperture must have the same length as list_ra and list_dec.")
        else:
            list_aperture = np.atleast_1d(aperture)
        # Load the tile data
        if not self.tbl_RIS:
            self.tbl_RIS = ascii.read(self.tile_path)
        if not self.coords_RIS:
            self.coords_RIS = SkyCoord(ra=self.tbl_RIS['ra'], dec=self.tbl_RIS['dec'], unit=(u.deg, u.deg))

        list_matched_tiles = []        
        list_matched_coords = []
        list_distance_to_boundary = []
        list_overlapped_areas = []

        for i, (ra, dec, aperture) in enumerate(zip(list_ra, list_dec, list_aperture)):
            target_point = Point(ra, dec)
            coord_targets = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))

            # Find nearby tiles
            nearby_tiles_idx = coord_targets.separation(self.coords_RIS) < (aperture + 2) * u.deg
            RIS_polygons_nearby = self._create_polygons(self.tbl_RIS[nearby_tiles_idx])

            if aperture == 0:  # Point-based matching
                closest_tile_id, distance_to_boundary = self._find_innermost_tile(polygons_by_id = RIS_polygons_nearby, target_point = target_point)
                if closest_tile_id is not None:
                    list_matched_tiles.append([closest_tile_id])
                    list_matched_coords.append(i)
                    list_distance_to_boundary.append([distance_to_boundary])
                    list_overlapped_areas.append([1])
            else:  # Aperture-based matching
                target_circle = target_point.buffer(aperture)
                overlapped_tiles, overlapped_areas = self._find_overlapped_tiles(polygons_by_id = RIS_polygons_nearby, target_circle = target_circle, fraction_overlap_lower = fraction_overlap_lower)
                if overlapped_tiles:
                    list_matched_tiles.extend([overlapped_tiles])
                    list_matched_coords.append(i)
                    list_distance_to_boundary.append([0.5]*len(overlapped_areas))
                    list_overlapped_areas.append(overlapped_areas)

        if not list_matched_tiles:
            return Table(), list_matched_coords, None

        if len(list_matched_tiles)!=len(list_ra):
            matched_set = set(list_matched_coords)
            all_indices = set(range(len(list_ra)))
            unmatched_indices = sorted(all_indices - matched_set)
            if unmatched_indices:
                unmatched_ra, unmatched_dec = [list_ra[i] for i in unmatched_indices], [list_dec[i] for i in unmatched_indices]
                print("No match:", ", ".join([f"{ra:.3f} {dec:.3f}" for ra, dec in zip(unmatched_ra, unmatched_dec)]))  

        matched_tbl = Table()
        for matched_coord, matched_tiles, distance_to_boundaries, overlapped_areas in zip(list_matched_coords, list_matched_tiles, list_distance_to_boundary, list_overlapped_areas):
            matched_tbl_single = self.tbl_RIS[np.isin(self.tbl_RIS['id'], matched_tiles)]
            matched_tbl_single['matched_idx'] = matched_coord
            matched_tbl_single['distance_to_boundary'] = distance_to_boundaries
            matched_tbl_single['overlapped_area'] = overlapped_areas
            matched_tbl_single['is_within_boundary'] = np.array(distance_to_boundaries) > match_tolerance_minutes/60
            matched_tbl = vstack([matched_tbl, matched_tbl_single])
        
        _, unique_indices = np.unique(matched_tbl['id'], return_index=True)
        unique_table = matched_tbl[sorted(unique_indices)]
        self._tile_table = unique_table

        # Build complete data first
        ras = []
        decs = []
        hhmmss_list = []
        ddmmss_list = []
        j2000_list = []
        apertures = []
        matched_indices = []
        matched_tiles_list = []

        for coord_idx, matched_tiles in zip(list_matched_coords, list_matched_tiles):
            ra = list_ra[coord_idx]
            dec = list_dec[coord_idx]
            ras.append(ra)
            decs.append(dec)
            
            hhmmss, ddmmss = parse_coordinates([ra,dec], output_format='hmsdms')
            hhmmss_list.append(hhmmss)
            ddmmss_list.append(ddmmss)
            
            j2000 = parse_coordinates([ra,dec], output_format='j2000')
            j2000_list.append(j2000)
            
            apertures.append(list_aperture[coord_idx])
            matched_indices.append(coord_idx)
            matched_tiles_list.append(matched_tiles)

        # Create table once with all data
        target_tbl = Table()
        target_tbl['ra'] = ras
        target_tbl['dec'] = decs
        target_tbl['hhmmss'] = hhmmss_list
        target_tbl['ddmmss'] = ddmmss_list
        target_tbl['j2000'] = j2000_list
        target_tbl['aperture'] = apertures
        target_tbl['matched_idx'] = matched_indices
        target_tbl['matched_tile'] = Column(matched_tiles_list, dtype=object)

        self._target_table = target_tbl
        
        fig_path = None
        
        if visualize:
            fig_path = self.visualize_tiles(
                **kwargs
            )

        return fig_path

    def visualize_tiles(self, 
                        visualize_ncols: int = 5, 
                        visualize_savepath: Union[str, bool] = False,
                        show: bool = True,
                        **kwargs):
        """
        Visualize the tiles and matched coordinates with aperture regions.
        Uses the target_table and tile_table properties directly.
        
        Parameters:
        - visualize_ncols (int): Number of columns in the visualization grid
        - visualize_savepath (str or bool): Path to save the visualization or False to not save
        - show (bool): Whether to show the plot or not
        
        Returns:
        - Path to the saved figure, or None if not saved
        """
        if not hasattr(self, '_target_table') or not hasattr(self, '_tile_table'):
            print("Target table or tile table not produced. Run find_overlapping_tiles first.")
            return None
        
        # Get unique matched indices
        unique_indices = np.unique(self.target_table['matched_idx'])
        n_coords = len(unique_indices)
        
        cols = visualize_ncols
        rows = (n_coords + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3), subplot_kw={'aspect': 'equal'})
        if rows == 1 and cols == 1:
            axes = [axes]
        elif isinstance(axes, np.ndarray):
            axes = axes.flatten()

        unmatched_j2000 = []
        
        for i, idx in enumerate(unique_indices):
            # Get target data
            target_row = self.target_table[self.target_table['matched_idx'] == idx][0]
            ra = target_row['ra']
            dec = target_row['dec']
            aperture = target_row['aperture']
            j2000 = target_row['j2000']
            
            # Get matched tile IDs for this target
            matched_tile_rows = self.tile_table[self.tile_table['matched_idx'] == idx]
            matched_tile_ids = matched_tile_rows['id']
            
            ax = axes[i]
            coord_targets = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))

            # Format J2000 string
            ra_sex = coord_targets.ra.to_string(unit=u.hour, sep='', precision=2, pad=True)
            dec_sex = coord_targets.dec.to_string(unit=u.deg, sep='', alwayssign=True, precision=2, pad=True)
            j2000_string = j2000 if j2000 else f"J{ra_sex}{dec_sex}"

            if len(matched_tile_ids) == 0:
                ax.scatter(ra, dec, color='gray', marker='x', s=60)
                ax.text(ra, dec, 'NO TILE', fontsize=8, ha='center', va='center', color='red',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
                ax.set_xlim(ra - 2, ra + 2)
                ax.set_ylim(dec - 2, dec + 2)
                if j2000:
                    unmatched_j2000.append(j2000_string)
                ax.set_title(f'Target {i + 1}: ({ra:.2f}, {dec:.2f})\n{j2000_string}')
                continue
            
            nearby_tiles_idx = coord_targets.separation(self.coords_RIS) < (aperture + 3) * u.deg
            nearby_polygons_by_id = self._create_polygons(self.tbl_RIS[nearby_tiles_idx])

            # Plot surrounding polygons in blue
            for tile_id, polygons in nearby_polygons_by_id.items():
                for poly in polygons:
                    x, y = poly.exterior.xy
                    ax.plot(x, y, color='blue', lw=1)
                    ax.fill(x, y, color='blue', alpha=0.3)

            # Plot matched polygons in red
            for tile_id in matched_tile_ids:
                if tile_id in nearby_polygons_by_id:
                    for poly in nearby_polygons_by_id[tile_id]:
                        x, y = poly.exterior.xy
                        ax.plot(x, y, color='red', lw=2)

            # Draw the aperture if applicable
            if aperture > 0:
                aperture_circle = plt.Circle((ra, dec), aperture, color='red', fill=True, linestyle='--', label='Aperture', lw=3, alpha=0.2)
                ax.add_patch(aperture_circle)
                ax.text(ra, dec, f'N_tiles ={len(matched_tile_ids)}', fontsize=8, ha='center', va='center', color='black',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
                ax.set_xlim(ra - 1.2 * aperture, ra + 1.2 * aperture)
                ax.set_ylim(dec - 1.2 * aperture, dec + 1.2 * aperture)
            else:
                ax.scatter(ra, dec, color='green', marker='o', s=50, label='Target Point' if i == 0 else None)
                ax.set_xlim(ra - 2, ra + 2)
                ax.set_ylim(dec - 2, dec + 2)
                if len(matched_tile_ids) > 0:
                    first_tile_id = matched_tile_ids[0]
                    if first_tile_id in nearby_polygons_by_id:
                        innermost_poly = nearby_polygons_by_id[first_tile_id][0]
                        x, y = innermost_poly.exterior.xy
                        ax.plot(x, y, color='red', lw=2)
                        ax.text(ra, dec - 0.3, first_tile_id, fontsize=8, ha='center', va='center', color='black',
                                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
            ax.set_title(f'Target {i + 1}: ({ra:.2f}, {dec:.2f})\n{j2000_string}')

        # Add a legend to the first subplot
        if n_coords > 0:
            axes[0].legend(loc='upper left', fontsize=8)
        
        # Add unmatched J2000 coords
        if unmatched_j2000:
            unmatched_label = "No match: " + ", ".join(unmatched_j2000)
            fig.text(0.5, 0.02, unmatched_label, ha='center', va='top', fontsize=10, color='red')
            fig.subplots_adjust(bottom=0.15)

        # Remove unused subplots
        for j in range(n_coords, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        fig_path = None
        if visualize_savepath:
            os.makedirs(visualize_savepath, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            fig_path = f"{os.path.join(visualize_savepath, f'matched_tiles_{timestamp}')}.png"
            plt.savefig(fig_path)
        
        if show:
            plt.show()
        else:
            plt.close()
        return fig_path

    def _find_innermost_tile(self, polygons_by_id, target_point):
        """
        Determine the innermost tile for a given target point.
        """
        max_distance = -float('inf')
        closest_tile_id = None

        for tile_id, polygons in polygons_by_id.items():
            for poly in polygons:
                if poly.contains(target_point) or poly.boundary.contains(target_point):
                    distance_to_boundary = target_point.distance(poly.boundary)
                    if distance_to_boundary > max_distance:
                        max_distance = distance_to_boundary
                        closest_tile_id = tile_id
        return closest_tile_id, max_distance

    def _find_overlapped_tiles(self, polygons_by_id, target_circle, fraction_overlap_lower = 0.2):
        """
        Find all tiles overlapping with the given target circle.
        """
        overlapped_tiles = []
        overlapped_area = []

        for tile_id, polygons in polygons_by_id.items():
            for poly in polygons:
                # Skip degenerate polygons with zero or near-zero area
                if poly.area < 1e-10:
                    continue
                    
                if target_circle.intersects(poly):
                    intersection = target_circle.intersection(poly)
                    fraction_overlap = intersection.area / poly.area
                    if fraction_overlap > fraction_overlap_lower:
                        overlapped_tiles.append(tile_id)
                        overlapped_area.append(fraction_overlap)
                    break  # Avoid duplicate entries for the same tile_id
        return overlapped_tiles, overlapped_area

    def _split_wrapping_polygon(self, polygon):
        """
        Splits a polygon into two parts if it crosses the RA = 0 (360) boundary.
        Ensures correct wrapping at the boundary and forms two valid polygons.

        Parameters:
        - polygon: Shapely Polygon object.

        Returns:
        - A list of Shapely Polygons split at the RA = 0 (360) boundary.
        """
        exterior_coords = list(polygon.exterior.coords)[:4]
        list_ra = [coord[0] for coord in exterior_coords]
        list_dec = [coord[1] for coord in exterior_coords]
        is_crossing = (np.max(list_ra) - np.min(list_ra)) > 180
        if not is_crossing:
            return [polygon]
        else:
            # Contruct polygons for the two parts
            left_part = []  # Coordinates for the part near RA = 0
            right_part = []  # Coordinates for the part near RA = 360
            left_part_idx = [0,1]
            right_part_idx = [2,3]
            for idx_left in left_part_idx:
                left_part.append(exterior_coords[idx_left])
            left_part.append([0, left_part[1][1]])
            left_part.append([0, left_part[0][1]])
            left_part.append(left_part[0])
            for idx_right in right_part_idx:
                right_part.append(exterior_coords[idx_right])
            right_part.append([360, right_part[1][1]])
            right_part.append([360, right_part[0][1]])
            right_part.append(right_part[0])
            # Create polygons
            polygons = []
            if len(left_part) > 2:
                polygons.append(Polygon(left_part))
            if len(right_part) > 2:
                polygons.append(Polygon(right_part))

            return polygons
        
    def _create_polygons(self, table):
        """
        Create polygons for the tiles, splitting them if necessary.
        """
        polygons_by_id = {}
        for row in table:
            try:
                corners = [
                    (float(row['ra1']), float(row['dec1'])),
                    (float(row['ra2']), float(row['dec2'])),
                    (float(row['ra3']), float(row['dec3'])),
                    (float(row['ra4']), float(row['dec4'])),
                ]
                polygons_by_id[row['id']] = self._split_wrapping_polygon(Polygon(corners))
            except Exception as e:
                print(f"Failed to create polygon for row {row}: {e}")
        return polygons_by_id
    
    def to_skycoord(self,
                    ra: Union[float, str, List[Union[float, str]]],
                    dec: Union[float, str, List[Union[float, str]]],
                    frame: str = 'icrs') -> SkyCoord:
        """
        Converts RA and Dec (single or list) to an Astropy SkyCoord object.
        """
        # Normalize to list
        if not isinstance(ra, list):
            ra = [ra]
        if not isinstance(dec, list):
            dec = [dec]

        if len(ra) != len(dec):
            raise ValueError("RA and Dec must have the same number of elements.")

        ra_strs = [str(r).strip() for r in ra]
        dec_strs = [str(d).strip() for d in dec]

        # Detect unit type
        sample_ra = ra_strs[0]
        sample_dec = dec_strs[0]
        if any(symbol in sample_ra for symbol in [':', 'h', ' ']) and any(symbol in sample_dec for symbol in [':', 'd', ' ']):
            units = (u.hourangle, u.deg)
        else:
            units = (u.deg, u.deg)

        return SkyCoord(ra=ra_strs, dec=dec_strs, unit=units, frame=frame)

# Example usage
if __name__ == "__main__":
    T = Tiles()
    list_ra = [173.6688]
    list_dec = [-21.05625]
    list_j2000 = ['J110633.45-182124.19']    
    tbl_filtered, tbl_idx, fig_path = T.find_overlapping_tiles(list_ra, list_dec, list_j2000,
                                                               list_aperture = 0, visualize=True, 
                                                               visualize_ncols=5, visualize_savepath='./output', 
                                                               match_tolerance_minutes=4, fraction_overlap_lower= 0.1 )
# %%