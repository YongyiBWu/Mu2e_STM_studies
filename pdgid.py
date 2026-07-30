from __future__ import print_function
import sys, os
import numpy as np

pdgid_dict = {
      11        : r'$e^{-}$',      # electron
    - 11        : r'$e^{+}$',      # positron
      13        : r'$\mu^{-}$',    # muon
    - 13        : r'$\mu^{+}$',    # positive muon
      22        : r'$\gamma$',     # photon
     111        : r'$\pi^{0}$',    # pi 0
     130        : r'$K_{L}^{0}$',  # K long
     211        : r'$\pi^{+}$',    # pi plus
    -211        : r'$\pi^{-}$',    # pi minus
     310        : r'$K^{0}_{S}$',  # K short
     321        : r'$K^{+}$',      # K plus
    -321        : r'$K^{-}$',      # K minus
     2112       : r'$n$',          # neutron
     2212       : r'$p$',          # proton
     3112       : r'$\Sigma^{-}$', # Sigma minus
     3122       : r'$\Lambda^{0}$',# Lambda 0
     3212       : r'$\Sigma^{0}$', # Sigma 0
     1000010020 : r'$D$',          # deuteron
     1000010030 : r'$T$',          # tritium
     1000020030 : r'$^{3}He$',     # helium-3
     1000020040 : r'$^{4}He$',     # helium-4
     1000080179 : r'$^{17}O^{*}$', # oxygen-17 excited
     1000110230 : r'$^{23}Na$',    # sodium-23
     1000120250 : r'$^{25}Mg$',    # magnesium-25
     1000120270 : r'$^{27}Mg$',    # magnesium-27
}

pdgid_color_dict = {
      11        : 'red',          # electron
    - 11        : 'green',        # positron
      13        : 'cyan',         # muon
    - 13        : 'gold',         # positive muon
      22        : 'magenta',      # photon
     111        : 'springgreen',  # pi 0
     130        : 'pink',         # K long
     211        : 'turquoise',    # pi plus
    -211        : 'lime',         # pi minus
     321        : 'chocolate',    # K plus
    -321        : 'salmon',       # K minus
     2112       : 'black',        # neutron
     2212       : 'blue',         # proton
     3112       : 'olive',        # Sigma minus
     3122       : 'orchid',       # Lambda 0
     3212       : 'powderblue',   # Sigma 0
     1000010020 : 'olivedrab',    # deuteron
     1000010030 : 'palegreen',    # tritium
     1000020030 : 'teal',         # helium-3
     1000020040 : 'navy',         # helium-4
     1000080179 : 'yellow',       # oxygen-17 excited
     1000110230 : 'plum',         # sodium-23
     1000120250 : 'deeppink',     # magnesium-25
     1000120270 : 'pink',         # magnesium-27
}
