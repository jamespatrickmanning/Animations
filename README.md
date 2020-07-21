This set of routines currently makes animations of doppio and gomofs model output for user specified geographic box and time

flowchart:https://www.draw.io/#G1ZhBB4jPLNwq5AAgtUTsrAXUDJoY1Tsja

Developed originally by JiM and his interns Lei Zhao and Mingchao in 2019-2020 but still needs a lot of cleaning, simplifying, and documenting.  For example, it calls a function called "seperate" which creates a dictionary of eMOLT observation positions to overlay on the images. This is not critical to making model animations and probably should be simplified and made to be optional.

In the Summer of 2020, Jack Polentes may be adding other features to the animation including a) current vectors and b) drifter tracks.

As of 13 July 2020,  JiM added a "get_doppio_tracks" routine which plots forecasted tracks but this file might be foundin other repositories to, for example, animate drifter tracks.
