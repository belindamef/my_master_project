import numpy as np
import copy as cp
import os.path
import pandas as pd
from PIL import Image
from psychopy import visual
from .node_to_rowcol import node_to_rowcol
from .rowcol_to_xy import rowcol_to_xy

# Directory management
working_dir = os.getcwd()  # working dir
project_dir = os.sep.join(working_dir.split(os.sep)[:4])
stimuli_dir = os.path.join(project_dir, 'code', 'stimuli')  # stimuli directory


class StimulusCreation:
    """Methods to create stimuli with the psychopy module to implement the
    treasure hunt task

    Attributes
    ----------
    todo
    """
    def __init__(self, task_params):
        """
        This function is the instantiation operation of the gridworld
        stimulus class, created with psychopy.visual.ElementArrayStim()

        Input
            dim       : dimensionality (No. of rows and columns of the gridworld
            gridsize  : size of the gridworld given in cm
            TODO

        Output
            gridstim : gridworld stimulus of class visual.ElementArrayStim
        """
        self.text_col = '#D6D6DB'
        self.back_col = '#1C1C1E'
        self.dim = task_params['dim']
        self.gridsize = task_params['gridsize']
        self.my_mac = task_params['my_mac']  # Monitor settings
        self.win = task_params['win']  # Window setting
        self.n_hides = task_params['n_hides']
        self.trials = task_params['trials']
        self.rounds = task_params['rounds']
        self.blocks = task_params['blocks']
        self.win.color = self.back_col
        self.cube_size = self.gridsize / self.dim
        #self.s1_t = task_params['s1_pos']
        #self.s7_tr_loc = task_params['s7_tr_loc'] # TODO: right condition that this is only used for figure creation

        # Initialize attributes (i.e. stimuli) that can be created in this class
        self.ready = None
        self.instr_top = None
        self.instr_center = None
        self.instr_low = None
        self.fix = None
        self.grid = None
        self.move_count = None
        self.round_count = None
        self.block_count= None
        self.score_count = None
        self.score_tr = None
        self.cube = None
        self.treasure = None
        self.drill = None
        self.startcube = None
        self.starttext = None
        self.current_pos = None
        self.endcube = None
        self.endtext = None
        self.arrow_right = None
        self.arrow_left = None
        self.arrow_up = None
        self.arrow_down = None
        self.pos_cross = None
        self.hides = {}

    def create_text_stims(self):
        """
        Create test stimuli
        """
        self.ready = visual.TextStim(self.win, text="Ready", height=0.8)

        self.starttext = visual.TextStim(self.win, text='START',
                                         height=0.5, bold=True,
                                         color='#067D39')

        self.current_pos = visual.TextStim(self.win, text='Current \n position',
                                           height=0.4, bold=True,
                                           color='#0000CC')
        grid_boarder = self.gridsize / 2
        self.instr_top = visual.TextStim(self.win, height=0.8, pos=[0, grid_boarder + 1.3], color=self.text_col)
        self.instr_center = visual.TextStim(self.win, height=0.8, color=self.text_col)
        self.instr_low = visual.TextStim(self.win, height=0.6, pos=[0, -(grid_boarder + 1.3)], color=self.text_col)
        self.fix = visual.TextStim(self.win, text="+", color=self.text_col)

    def create_figure_stims(self):
        """
        This function creates the gridworld stimulus

        Input
            dim       : dimensionality (No. of rows and columns of the gridworld
            gridsize  : size of the gridworld given in cm
            TODO

        Output
            gridstim : gridworld stimulus of class visual.ElementArrayStim
        """

        # Create array for gridline positions, evenly spread along x/y-axis relative to gridsize
        # e.g. for dim = 5: [-5., -3., -1.,  1.,  3.,  5.]
        line_pos = np.linspace(0, self.gridsize, self.dim + 1) - self.gridsize / 2

        # Create array with x/y-axis positions of the center of each gridline relative to the field center
        # e.g. dim = 5: [[0.,-5.], [0.,-3.], [0.,-1.], [0.,1.], [0.,3.], [0.,5.],
        #               [-5.,0.], [-3.,0.], [-1.,0.], [1.,0.], [3.,0.], [5.,0.]])
        gridline_xys = np.column_stack((np.concatenate((np.zeros(self.dim + 1), line_pos)),
                                        np.concatenate((line_pos, np.zeros(self.dim + 1)))))

        # Create array with orientations for each gridline
        # e.g. for dim = 5: [0, 0, 0, 0, 0, 0, 90, 90, 90, 90, 90, 90]
        # An ori of 0 is vertical; increasing ori values are increasingly clockwise
        # CAVE, here it's somehow the other way around
        line_oris = np.concatenate((np.zeros(self.dim + 1), np.full(self.dim + 1, 90)))

        # Create gridworld stimulus
        self.grid = visual.ElementArrayStim(self.win, fieldPos=(0.0, 0.0),
                                            nElements=(self.dim + 1) * 2,
                                            sizes=[self.gridsize, 0.03], xys=gridline_xys,
                                            colors='#C0C0C0', oris=line_oris, sfs=0,
                                            elementTex=None, elementMask=None, texRes=48)

        # Create arrows half the length of one node field
        cube_size = cp.deepcopy(self.cube_size)
        arrow_right_vert = [(-cube_size / 2, cube_size / 8), (-cube_size / 2, -cube_size / 8),
                            (-cube_size / 4, -cube_size / 8), (-cube_size / 4, -cube_size / 4),
                            (0, 0),
                            (-cube_size / 4, cube_size / 4), (-cube_size / 4, cube_size / 8)]
        arrow_left_vert = [(cube_size / 2, cube_size / 8), (cube_size / 2, -cube_size / 8),
                           (cube_size / 4, -cube_size / 8), (cube_size / 4, -cube_size / 4),
                           (0, 0),
                           (cube_size / 4, cube_size / 4), (cube_size / 4, cube_size / 8)]
        arrow_up_vert = [(-cube_size / 8, -cube_size / 2), (cube_size / 8, -cube_size / 2),
                         (cube_size / 8, -cube_size / 4), (cube_size / 4, -cube_size / 4),
                         (0, 0),
                         (-cube_size / 4, -cube_size / 4), (-cube_size / 8, -cube_size / 4)]
        arrow_down_vert = [(-cube_size / 8, cube_size / 2), (cube_size / 8, cube_size / 2),
                           (cube_size / 8, cube_size / 4), (cube_size / 4, cube_size / 4),
                           (0, 0),
                           (-cube_size / 4, cube_size / 4), (-cube_size / 8, cube_size / 4)]
        self.arrow_right = visual.ShapeStim(self.win, vertices=arrow_right_vert,
                                            fillColor='#067D39', lineColor='#067D39',
                                            opacity=0.5)
        self.arrow_left = visual.ShapeStim(self.win, vertices=arrow_left_vert,
                                           fillColor='#067D39', lineColor='#067D39',
                                           opacity=0.5)
        self.arrow_up = visual.ShapeStim(self.win, vertices=arrow_up_vert,
                                         fillColor='#067D39', lineColor='#067D39',
                                         opacity=0.5)
        self.arrow_down = visual.ShapeStim(self.win, vertices=arrow_down_vert,
                                           fillColor='#067D39', lineColor='#067D39',
                                           opacity=0.5)
        self.pos_cross = visual.TextStim(self.win, text='X',
                                         height=cube_size, bold=True,
                                         color='#B20E0E')
        # Create cube of the size of one node field
        self.cube = visual.Rect(self.win, lineColor=None, fillColor='white',
                                size=self.cube_size * 0.7, ori=45, opacity=0.8)

        # Create cube animated drilling stimulus
        self.drill = visual.Rect(self.win, lineColor=None, fillColor='white',
                                 size=self.cube_size * 2)#, opacity=0.8)

    def create_count_stims(self):
        """
        Create stimuli that prompt move, round and score counts
        """
        grid_boarder = self.gridsize / 2  # gridworld border: left (neg), right (pos),
                                          # lower (neg), upper (pos)
                                          # needed for text positions
        self.move_count = visual.TextStim(self.win,
                                          height=0.8,
                                          pos=[- grid_boarder, grid_boarder - 1],
                                          color=self.text_col, alignText='left')
        self.round_count = visual.TextStim(self.win, height=0.8,
                                           pos=[- grid_boarder, grid_boarder - 3],
                                           color=self.text_col, alignText='left')
        self.block_count = visual.TextStim(self.win, height=0.8,
                                           pos=[- grid_boarder, grid_boarder - 4],
                                           color=self.text_col, alignText='left')
        self.score_count = visual.TextStim(self.win, text='Total Score:',
                                           height=0.8, color=self.text_col,
                                           pos=[-grid_boarder, grid_boarder - 5],
                                           alignText='left')

    def create_score_tr_stim(self, score=0):
        """
        Create stimulus that visualizes current score count.

        Parameters
        ----------
        score : int, optional
            score count, number of treasures to be displayed
        """
        grid_boarder = self.gridsize / 2

        if score > 0:
            tr_count_image_filename = f"{stimuli_dir}/score_prompt_blocks-{self.blocks}_" \
                                      f"rounds-{self.rounds}/{score}_treasures.png"

            tr_count_image = Image.open(tr_count_image_filename)
            self.score_tr = visual.ImageStim(self.win, image=tr_count_image_filename,
                                             pos=[-grid_boarder - 5, grid_boarder - 8],
                                             size=4)
        else:
            self.score_tr = visual.TextStim(self.win, text='0',
                                            height=0.8, color=self.text_col,
                                            pos=[- grid_boarder, grid_boarder - 7],
                                            alignText='left')

    def create_treasure_stim(self):
        """
        Create treasure picture stimulus
        """
        treasure_image = os.path.join(stimuli_dir, 'treasure.png')
        self.treasure = visual.ImageStim(self.win, image=treasure_image, size=self.cube_size)

    def create_hides_stims(self, s4_hide_node):
        """
        Create stimulus that highlight unveiled hiding spots

        Parameters
        ----------
        s4_hide_node
        todo
        """
        for node, value in enumerate(s4_hide_node):
            if value == 0:
                self.hides[node] = visual.Rect(
                    self.win, lineColor=None, fillColor='#C0C0C0', size=self.cube_size, opacity=0.5,
                    pos=rowcol_to_xy(node_to_rowcol(np.array(node), self.dim), self.dim, self.gridsize))
            elif value == 1:
                self.hides[node] = visual.Rect(
                    self.win, lineColor=None, fillColor='#006666', size=self.cube_size, opacity=0.5,
                    pos=rowcol_to_xy(node_to_rowcol(np.array(node), self.dim), self.dim, self.gridsize))

    def create_pos_stim_for_fig(self, block, hround):
        """
        This function creates the gridworld stimulus

        Input
            dim       : dimensionality (No. of rows and columns of the gridworld
            gridsize  : size of the gridworld given in cm
            TODO

        Output

        """
        # Create position stimuli
        # -----------------------------------------------------------------------------
        C = self.rounds  # Number of rounds
        T = self.trials  # Number of trials
        s_1_t = self.s1_t
        s_7_c = self.s7_tr_loc
        starttrial_c = hround * T + block * C * T  # First this_trial of current hunting round
        end_index_c = int(pd.DataFrame(s_1_t[starttrial_c:starttrial_c + T]).apply(pd.Series.last_valid_index).values)
        endtrial_c = (starttrial_c + end_index_c)

        # Get starting end ending position of current round
        startpos_c = np.array(s_1_t[starttrial_c])  # start
        startpos_c_rowcol = node_to_rowcol(startpos_c, self.dim)  # Transform to rowcol
        endpos_c = s_1_t[endtrial_c]  # end
        endpos_c_rowcol = node_to_rowcol(endpos_c, self.dim)  # Transform to rowcol

        # Get treasure position of current round
        th_c = np.array(s_7_c[hround * T])
        th_c_rowcol = node_to_rowcol(th_c, self.dim)

        # Create starting end ending position stimuli
        startpos_c_xy = rowcol_to_xy(startpos_c_rowcol, self.dim, self.gridsize)  # Transform to xy
        endpos_c_xy = rowcol_to_xy(endpos_c_rowcol, self.dim, self.gridsize)  # Transform to xy

        # Create cube stimuli
        self.startcube = visual.Rect(self.win, lineColor=None, fillColor='white',
                                     pos=startpos_c_xy, size=self.cube_size, opacity=0.8)

        self.starttext = visual.TextStim(self.win, text='START',
                                         height=0.4, bold=True,
                                         pos=startpos_c_xy,
                                         color='#067D39')

        self.endcube = visual.Rect(self.win, lineColor=None, fillColor='white',
                                   pos=endpos_c_xy, size=self.cube_size, opacity=0.6)

        self.endtext = visual.TextStim(self.win, text='END',
                                       height=0.6, bold=True,
                                       pos=endpos_c_xy,
                                       color='#990000')

        # Create treasure location position stimuli
        treasure_pos_c_xy = rowcol_to_xy(th_c_rowcol, self.dim, self.gridsize)  # Transform to xy
        treasure_image = os.path.join(stimuli_dir, 'treasure.png')
        self.treasure = visual.ImageStim(self.win, image=treasure_image,
                                         size=self.cube_size * 0.5,
                                         pos=treasure_pos_c_xy)

    def create_stimuli(self):
        """Create stimuli for treasure hunt task implementation

        Parameter
        ---------
        task_params : dict todo
        """
        self.create_text_stims()
        self.create_figure_stims()
        self.create_count_stims()
        self.create_score_tr_stim()  # Initial score = O
        self.create_treasure_stim()

        # Create score count stimuli if not existent for this setting (
        # no of blocks and rounds)
        if not os.path.exists(
                f"{stimuli_dir}/\"score_prompt_blocks-{self.blocks}"
                f"_rounds-{self.rounds}"):
            self.create_score_prompt_png()

    # -----------------------------------------------------------------------------
    # Create stimuli png files
    # -----------------------------------------------------------------------------

    def create_grid_png(self):
        """This function presents the gridworld stimulus and takes a screenshot
        of it"""
        self.grid.draw()
        self.win.flip()
        self.win.getMovieFrame()
        self.win.saveMovieFrames(f"{stimuli_dir}/gridworld.png")

    def create_score_prompt_png(self):
        """This function creates score count stimuli for different numbers of treasures"""

        # Directory management
        tr_im_path = os.path.join(stimuli_dir, "treasure.png")
        tr_score_dir = os.path.join(stimuli_dir, "score_prompt_"
                                    f"blocks-{self.blocks}_"
                                    f"rounds-{self.rounds}")
        # Create directory if not existent
        if not os.path.exists(tr_score_dir):
            os.makedirs(tr_score_dir)

        # Calculate maximum number of treasures
        n_treasures = self.rounds * self.blocks

        # Initialize number of rows and columns for treasure pictures
        n_columns = 4
        if n_treasures % n_columns == 0:
            n_rows = int(n_treasures / n_columns)
        else:
            n_rows = int(n_treasures // n_columns + 1)

        for tr_score in range(1, n_treasures + 1):
            tr_file_names = []
            for i in range(tr_score):
                tr_file_names.append(f"{tr_im_path}")
            tr_figures = [Image.open(x) for x in tr_file_names]
            widths, heights = zip(*(i.size for i in tr_figures))
            # Create new figure with one treasure for each score
            total_width = n_columns * max(widths)
            total_height = n_rows * max(heights)
            # Determine rows to fill dependent on number of treasures
            if tr_score % n_columns == 0:
                n_rowstofill = int(tr_score / n_columns)
            else:
                n_rowstofill = int(tr_score // n_columns + 1)
            # Create new image, in which all treasure images will be pasted
            score_tr_image = Image.new('RGB', (total_width, total_height), color=self.back_col)
            # Start at y-axis offset 0
            y_offset = 0
            # Loop through rows
            for row in range(0, n_rowstofill):
                x_offset = 0
                for figure in tr_figures[(row * n_columns):(row * n_columns + n_columns)]:
                    score_tr_image.paste(figure, (x_offset, y_offset))
                    x_offset += figure.size[0]
                y_offset += max(heights)
            score_tr_image.save(f"{tr_score_dir}/{tr_score}_treasures.png")
