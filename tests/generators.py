# egegsignals - Software for processing electrogastroenterography signals.

# Copyright (C) 2013 -- 2018 Aleksandr Popov

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np


def harmonic(T, dt, f, A=1, phi=0):
    """Generate harmonis signal."""
    t = np.linspace(0, T, T/dt + 1)
    x = A * np.cos(2 * np.pi * f * t + phi)
    return (t, x)
