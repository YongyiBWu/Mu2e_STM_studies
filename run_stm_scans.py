#!/usr/bin/env python3
"""Scan definitions shared by the STMDet driver and the analysis scripts.

The data layout this encodes
----------------------------
A CAMPAIGN (stmedep00, stmedep01, ...) is one production run at ONE FIXED configuration.
Its catalogs live in

    /exp/mu2e/data/users/<user>/<campaign>/datasets/stmedep/catalog/
        dts.<user>.stmedeps20r0000.<campaign>.art.files    Ele
        dts.<user>.stmedeps20r0001.<campaign>.art.files    Mu
        dts.<user>.stmedeps20r0002.<campaign>.art.files    1809
        dts.<user>.stmedeps20r0003.<campaign>.art.files    Neutrals

so the only thing varying inside a campaign is the SOURCE. The configuration is a property
of the campaign as a whole and appears nowhere in the file names.

A SCAN is therefore a SET OF CAMPAIGNS - one per configuration - plus the variable values
each campaign stands for. Scanning SSC aperture means running stmedep00, stmedep01, ... at
different apertures and declaring here which is which. Nothing in the data records that
mapping, so this file is its only home: get a campaign's values wrong here and every plot
silently puts a correct measurement at the wrong x.

Variables, not a bare label
---------------------------
A campaign's configuration is a dict of NAMED variables with values, e.g.
    {'ssc_area': 50.0}                      a point in a 1-D aperture scan
    {'ssc_area': 50.0, 'hpge_xoff': -10.0}  a point in a 2-D scan
rather than a single opaque label. The analysis asks for a variable BY NAME to put on the x
axis, and whatever else varies becomes series or is held fixed - so adding a second scan
variable needs no change to the plotting code. Each variable carries its own axis label and
unit, so plots are annotated from these definitions rather than from literals scattered
through the scripts.

Output naming
-------------
Output files carry the campaign, since that is what identifies the configuration:
    STMDet_tree.<source>_<campaign>.root
and parse_output_path() turns such a path back into (source, CampaignPoint).
"""

import os
import re
import numpy as np

class Variable:
    """One scanned quantity: how to name it on an axis, and in what unit.

    `fmt` formats a value for a legend entry or a title; it is separate from the axis label
    because a legend wants "50 mm^2" while an axis wants "SSC upstream area [mm^2]".
    """

    def __init__(self, key, label, unit='', fmt='{:g}'):
        self.key = key          # e.g. 'ssc_area'
        self.label = label      # e.g. 'SSC upstream area'
        self.unit = unit        # e.g. 'mm^2'
        self.fmt = fmt

    def axis_label(self):
        return f'{self.label} [{self.unit}]' if self.unit else self.label

    def value_label(self, value):
        s = self.fmt.format(value)
        return f'{s} {self.unit}' if self.unit else s

    def __repr__(self):
        return f'Variable({self.key!r})'


class CampaignPoint:
    """One campaign and the configuration it was run at.

    `campaign` is both the directory name and the field in every catalog name, so it is the
    identity of this point everywhere. `note` is free text for whatever the variables do not
    capture (a geometry file version, say) - it appears in listings only.
    """

    def __init__(self, campaign, values, note=''):
        self.campaign = campaign
        self.values = dict(values)
        self.note = note

    def get(self, key):
        return self.values.get(key)

    def __repr__(self):
        return f'CampaignPoint({self.campaign!r}, {self.values!r})'


class Scan:
    """A set of campaigns differing in configuration, plus how their files are named.

    catalog_pattern and output_pattern are format strings over {user}, {dataset},
    {campaign}, {source}. Keeping both here - rather than in the driver - is what lets an
    analysis script recover a campaign from a path without knowing how the driver was run.
    """

    def __init__(self, name, variables, points,
                 data_root='/exp/mu2e/data/users/yongyiwu',
                 output_root='/exp/mu2e/data/users/yongyiwu/stm_analysis/edep',
                 catalog_subdir='datasets/stmedep/catalog',
                 catalog_pattern='dts.{user}.{dataset}.{campaign}.art.files',
                 output_pattern='STMDet_tree.{source}_{campaign}.root',
                 user='yongyiwu'):
        self.name = name
        self.variables = {v.key: v for v in variables}
        self.points = list(points)
        self.data_root = data_root
        self.output_root = output_root
        self.catalog_subdir = catalog_subdir
        self.catalog_pattern = catalog_pattern
        self.output_pattern = output_pattern
        self.user = user

        # A campaign appearing twice would make the path->point lookup ambiguous, and the
        # symptom would be two configurations silently collapsing into one on a plot.
        seen = {}
        for p in self.points:
            if p.campaign in seen:
                raise ValueError(
                    f'Scan {name!r}: campaign {p.campaign!r} listed twice. A campaign is one '
                    f'fixed configuration, so it belongs to exactly one point.')
            seen[p.campaign] = p
            unknown = set(p.values) - set(self.variables)
            if unknown:
                raise ValueError(
                    f'Scan {name!r}, campaign {p.campaign!r}: value(s) for undeclared '
                    f'variable(s) {sorted(unknown)}. Add a Variable for each, so the plots '
                    f'know how to label them.')
            missing = set(self.variables) - set(p.values)
            if missing:
                raise ValueError(
                    f'Scan {name!r}, campaign {p.campaign!r}: no value for '
                    f'{sorted(missing)}. Every point must set every declared variable, '
                    f'otherwise a plot cannot tell "held fixed" from "unknown".')
        self._by_campaign = seen

    # ---- name building ----

    def catalog_dir(self, campaign):
        return os.path.join(self.data_root, campaign, self.catalog_subdir)

    def catalog_path(self, dataset, campaign):
        return os.path.join(self.catalog_dir(campaign), self.catalog_pattern.format(
            user=self.user, dataset=dataset, campaign=campaign))

    def output_dir(self, output_root=None):
        # One directory per SCAN, not per campaign: the campaign is already in every file
        # name, and a flat directory is what lets an analysis glob one scan in one pass.
        return os.path.join(output_root or self.output_root, self.name)

    def output_path(self, source, campaign, output_root=None):
        return os.path.join(self.output_dir(output_root), self.output_pattern.format(
            source=source, campaign=campaign))

    # ---- name parsing ----

    def _output_regex(self):
        """Turn output_pattern into a regex capturing source and campaign.

        Built from the same pattern used to WRITE the names, so the two cannot drift.

        The campaign group is an ALTERNATION OVER THIS SCAN'S KNOWN CAMPAIGNS rather than a
        wildcard. The separator between {source} and {campaign} is '_', which a source label
        may itself contain ('TargetStops_1809'), and a wildcard campaign group would then
        split at the wrong underscore: source='TargetStops', campaign='1809_stmedep00',
        which no campaign matches, so parse_output_path would return (None, None) and the
        analysis would silently drop those files. Matching the campaign against the names
        that actually exist removes the ambiguity entirely. Longest-first so a campaign name
        that prefixes another cannot win on a shorter match.
        """
        campaigns = sorted((p.campaign for p in self.points), key=len, reverse=True)
        campaign_group = ('(?P<campaign>' + '|'.join(re.escape(c) for c in campaigns) + ')'
                          if campaigns else r'(?P<campaign>[^.]+?)')
        pattern = re.escape(self.output_pattern)
        pattern = pattern.replace(re.escape('{source}'), r'(?P<source>[^.]+?)')
        pattern = pattern.replace(re.escape('{campaign}'), campaign_group)
        return re.compile('^' + pattern + '$')

    def parse_output_path(self, path):
        """(source, CampaignPoint) for a file this scan produced, or (None, None).

        Returning None rather than raising lets a caller sweep a directory holding several
        scans' outputs and skip what is not this scan's; a caller that expects a match should
        treat None as an error rather than a default.
        """
        m = self._output_regex().match(os.path.basename(path))
        if not m:
            return (None, None)
        point = self._by_campaign.get(m.group('campaign'))
        if point is None:
            return (None, None)
        return (m.group('source'), point)

    # ---- queries the analysis uses ----

    def variable(self, key):
        if key not in self.variables:
            raise KeyError(
                f'Scan {self.name!r} has no variable {key!r}; it varies '
                f'{sorted(self.variables)}.')
        return self.variables[key]

    def varying(self):
        """Variable keys that actually take more than one value across the campaigns.

        A 2-D scan run as a 1-D slice (one variable held fixed) reports only the one that
        moves, so a plot's x axis is chosen from what the data supports rather than from
        what the scan declares.
        """
        return [key for key in self.variables
                if len({p.get(key) for p in self.points}) > 1]

    def points_sorted_by(self, key):
        """Campaigns ordered by one variable - the x ordering of a scan plot.

        For a CATEGORICAL variable (string values, e.g. physics_list) this sorts
        alphabetically, which is an arbitrary order carrying no physical meaning. Such a
        variable belongs on a bar chart or as a set of series, with the category order
        chosen deliberately, not on a line plot against this ordering.
        """
        self.variable(key)   # raises with a useful message if the key is wrong
        values = [p.get(key) for p in self.points]
        # Mixed types would make the sort raise deep inside, with no hint of which scan or
        # variable was at fault; say so here instead.
        if len({type(v) for v in values}) > 1:
            raise TypeError(
                f'Scan {self.name!r}, variable {key!r}: campaigns give values of mixed types '
                f'{sorted({type(v).__name__ for v in values})}. Every campaign must give it '
                f'the same kind of value.')
        return sorted(self.points, key=lambda p: p.get(key))

    def is_categorical(self, key):
        """True when this variable's values are strings rather than numbers.

        The plotting code uses it to choose a bar/series presentation over a line, since
        there is no meaningful distance between two physics lists.
        """
        self.variable(key)
        return any(isinstance(p.get(key), str) for p in self.points)

    def campaigns(self):
        return [p.campaign for p in self.points]


# ---------------------------------------------------------------------------
# Source datasets. The labels are in dataSourceNames() order, so the index matches the
# summaryIndex the generator wrote and the r000N suffix in the dataset tag. These are the
# same four in every campaign.
# ---------------------------------------------------------------------------
SOURCES = [
    ('Ele',      'stmedeps20r0000'),
    ('Mu',       'stmedeps20r0001'),
    ('1809',     'stmedeps20r0002'),
    ('Neutrals', 'stmedeps20r0003'),
]


# ---------------------------------------------------------------------------
# The scans. Adding a configuration means producing a new campaign and adding one
# CampaignPoint here; neither the driver nor any analysis script needs editing.
#
# EVERY VALUE BELOW IS AN ASSERTION ABOUT HOW A CAMPAIGN WAS GENERATED, and nothing in the
# data can confirm it. Check each against the geometry the campaign actually ran before
# trusting a plot that uses it.
# ---------------------------------------------------------------------------

OLD_GEOM_SSC_AREA = Variable('old_geom_ssc_area', 'v09 SSC (upstream) area', 'mm^{2}')
OLD_GEOM_PHYSICS_LIST = Variable('old_geom_physics_list', 'Geant4 physics list', fmt='{}')
SSC_AREA = Variable('ssc_area', 'SSC (upstream) area', 'mm^{2}')
SSC_AREA_EXPANDED = Variable('ssc_area_expanded', 'SSC area (same upstream/downstream)', 'mm^{2}')
HPGE_XOFF = Variable('hpge_xoff', 'HPGe x offset', 'mm', fmt='{:+g}')
BRICK_GAP_HOR = Variable('brick_gap_hor', 'Lead brick gap (horizontal)', 'mm')
BRICK_GAP_VER = Variable('brick_gap_ver', 'Lead brick gap (vertical)', 'mm')
PHYSICS_LIST = Variable('physics_list', 'Geant4 physics list', fmt='{}')

SCANS = {
    'old_geom_ssc_area': Scan(
        name='old_geom_ssc_area',
        variables=[OLD_GEOM_SSC_AREA],
        points=[
            CampaignPoint('stmedep00', {'old_geom_ssc_area': 50.0}),
            CampaignPoint('stmedep01', {'old_geom_ssc_area': 100.0}),
            CampaignPoint('stmedep02', {'old_geom_ssc_area': 200.0}),
            CampaignPoint('stmedep03', {'old_geom_ssc_area': 400.0}),
            CampaignPoint('stmedep04', {'old_geom_ssc_area': 0.0}),
        ],
    ),

    'old_geom_physics_list': Scan(
        name='old_geom_physics_list',
        variables=[OLD_GEOM_PHYSICS_LIST],
        points=[
            # The existing EMZ campaign, reused as the baseline arm - no re-simulation
            # needed. Its SSC area must match the ShieldingM campaign's.
            CampaignPoint('stmedep00',
                          {'old_geom_physics_list': 'QGSP_BERT_EMZ'},
                          note='nonIonE is identically 0 - no HP neutron transport'),
            # The new campaign: same geometry, ShieldingM.
            CampaignPoint('stmedep05',
                          {'old_geom_physics_list': 'ShieldingM'},
                          note='Mu2e default list; HP neutrons, so nonIonE is populated'),
        ],
    ),

    'ssc_area': Scan(
        name='ssc_area',
        variables=[SSC_AREA],
        points=[
            CampaignPoint('stmedep10', {'ssc_area': 12.7**2*np.pi()}), # 500
            CampaignPoint('stmedep11', {'ssc_area': 6.985**2*np.pi()}), # 150
            CampaignPoint('stmedep12', {'ssc_area': 5.715**2*np.pi()}), # 100
            CampaignPoint('stmedep13', {'ssc_area': 3.9878**2*np.pi()}), # 50
            CampaignPoint('stmedep14', {'ssc_area': 2.8194**2*np.pi()}), # 25
            CampaignPoint('stmedep15', {'ssc_area': 0.001**2*np.pi()}), # 0
        ],
    ),

    'ssc_area_expanded': Scan(
        name='ssc_area_expanded',
        variables=[SSC_AREA_EXPANDED],
        points=[
            CampaignPoint('stmedep11', {'ssc_area_expanded': 6.985**2*np.pi()}), # 150
            CampaignPoint('stmedep16', {'ssc_area_expanded': 7.5692**2*np.pi()}), # 180
            CampaignPoint('stmedep17', {'ssc_area_expanded': 8.176**2*np.pi()}), # 210
            CampaignPoint('stmedep18', {'ssc_area_expanded': 8.921**2*np.pi()}), # 250
            CampaignPoint('stmedep19', {'ssc_area_expanded': 9.772**2*np.pi()}), # 300
            CampaignPoint('stmedep20', {'ssc_area_expanded': 11.284**2*np.pi()}), # 400
        ],
    ),

    'hpge_xoff': Scan(
        name='hpge_xoff',
        variables=[HPGE_XOFF],
        points=[
            CampaignPoint('stmedep10', {'hpge_xoff': -10.0}),
            CampaignPoint('stmedep21', {'hpge_xoff': -5.0}),
            CampaignPoint('stmedep22', {'hpge_xoff':  0.0}),
            CampaignPoint('stmedep23', {'hpge_xoff':  5.0}),
            CampaignPoint('stmedep24', {'hpge_xoff':  10.0}),
            CampaignPoint('stmedep25', {'hpge_xoff':  15.0}),
            CampaignPoint('stmedep26', {'hpge_xoff':  20.0}),
            CampaignPoint('stmedep27', {'hpge_xoff':  25.0}),
        ],
    ),

    'gap_ver_x_hor': Scan(
        name='gap_ver_x_hor',
        variables=[BRICK_GAP_VER, BRICK_GAP_HOR],
        points=[
            CampaignPoint('stmedep10', {'brick_gap_ver': 2*0.1, 'brick_gap_hor': 2*0.8}),
            CampaignPoint('stmedep28', {'brick_gap_ver': 2*0.1, 'brick_gap_hor': 2*0.4}),
            CampaignPoint('stmedep29', {'brick_gap_ver': 2*0.1, 'brick_gap_hor': 2*1.2}),
            CampaignPoint('stmedep30', {'brick_gap_ver': 2*0.2, 'brick_gap_hor': 2*0.4}),
            CampaignPoint('stmedep31', {'brick_gap_ver': 2*0.2, 'brick_gap_hor': 2*0.8}),
            CampaignPoint('stmedep32', {'brick_gap_ver': 2*0.2, 'brick_gap_hor': 2*1.2}),
        ],
    ),

    'physics_list': Scan(
        name='physics_list',
        variables=[PHYSICS_LIST],
        points=[
            # The standard EMZ campaign, reused as the baseline arm - no re-simulation
            # needed. Its SSC area must match the ShieldingM campaign's.
            CampaignPoint('stmedep10',
                          {'physics_list': 'QGSP_BERT_EMZ'},
                          note='nonIonE is identically 0 - no HP neutron transport'),
            # same geometry, ShieldingM.
            CampaignPoint('stmedep33',
                          {'physics_list': 'ShieldingM'},
                          note='Mu2e default list; HP neutrons, so nonIonE is populated'),
        ],
    ),
}

DEFAULT_SCAN = 'old_geom_ssc_area'


def get_scan(name=None):
    name = name or DEFAULT_SCAN
    if name not in SCANS:
        raise SystemExit(f'Unknown scan {name!r}. Defined: {", ".join(sorted(SCANS))}')
    scan = SCANS[name]
    if not scan.points:
        raise SystemExit(
            f'Scan {name!r} has no campaigns yet - it is a template. Add CampaignPoint '
            f'entries in run_stm_scans.py once the productions exist.')
    return scan


if __name__ == '__main__':
    # Printing the definitions is the cheapest way to check a scan before running it: it
    # exercises the duplicate-campaign and variable-coverage checks in Scan.__init__.
    for name, scan in sorted(SCANS.items()):
        print(f'{name}')
        print('  variables : ' + ', '.join(
            v.axis_label() for v in scan.variables.values()))
        if not scan.points:
            print('  points    : (none - template)\n')
            continue
        print(f'  varying   : {", ".join(scan.varying()) or "(nothing - single campaign)"}')
        print(f'  campaigns : {len(scan.points)}')
        for p in scan.points:
            vals = ', '.join(f'{k}={scan.variable(k).value_label(v)}'
                             for k, v in sorted(p.values.items()))
            tail = f'   # {p.note}' if p.note else ''
            print(f'    {p.campaign:12s} {vals}{tail}')
        first = scan.points[0]
        src0, ds0 = SOURCES[0]
        print(f'  catalog   : {scan.catalog_path(ds0, first.campaign)}')
        example = scan.output_path(src0, first.campaign)
        src, pt = scan.parse_output_path(example)
        print(f'  output    : {example}')
        print(f'  roundtrip : source={src}, campaign={pt.campaign if pt else None}')
        print()
