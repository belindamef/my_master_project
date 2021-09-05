from .abm_structure import AbmStructure  # Python structure simulation utility


def define_agent_model(mod):
    """
    This function returns the ABM framework task, agent, model initialization
    structures or model optimization parameters for a given agent-behavioral
    model

    Input
        mod : model specification structure with fields
                .model          : model index string
                .mod        : mode 'validation', 'simulation', or 'estimation'

            if mod == 'simulation'
                .theta      : model parameter values

    Output

    """
    # ---------------------------------------------------------------------
    # Random choice models
    # ---------------------------------------------------------------------
    if mod.model == 'C1':

        if mod.mode == 'simulation':
            mod.a_init = AbmStructure()  # agent initialization structure
            mod.a_init.agent = 'C1'  # agent index
            mod.a_init.bayesian_agent = False
            mod.a_init.exploring_agent = False
            mod.m_init = AbmStructure()  # model initialization structure

    # ---------------------------------------------------------------------
    # Random exploiter
    # ---------------------------------------------------------------------
    elif mod.model == 'C2':

        if mod.mode == 'simulation':
            mod.a_init = AbmStructure()  # agent initialization structure
            mod.a_init.agent = 'C2'  # agent index
            mod.a_init.bayesian_agent = False
            mod.a_init.exploring_agent = False

    # ---------------------------------------------------------------------
    # 50% random exploit and 50% random explore model
    # ---------------------------------------------------------------------
    elif mod.model == 'C3':

        if mod.mode == 'simulation':
            mod.a_init = AbmStructure()
            mod.a_init.agent = 'C3'
            mod.a_init.bayesian_agent = False
            mod.a_init.exploring_agent = False

    # ---------------------------------------------------------------------
    # Pure exploiter model
    # ---------------------------------------------------------------------
    elif mod.model == 'A1':

        if mod.mode == 'simulation':
            mod.a_init = AbmStructure()
            mod.a_init.agent = 'A1'
            mod.a_init.bayesian_agent = True
            mod.a_init.exploring_agent = False

    # ---------------------------------------------------------------------
    # Pure explorer and random exploit model
    # ---------------------------------------------------------------------
    elif mod.model == 'A2':

        if mod.mode == 'simulation':
            mod.a_init = AbmStructure()
            mod.a_init.agent = 'A2'
            mod.a_init.bayesian_agent = True
            mod.a_init.exploring_agent = True

    # ---------------------------------------------------------------------
    # Pure explorer and random exploit model
    # ---------------------------------------------------------------------
    elif mod.model == 'A3':

        if mod.mode == 'simulation':
            mod.a_init = AbmStructure()
            mod.a_init.agent = 'A3'
            mod.a_init.bayesian_agent = True
            mod.a_init.exploring_agent = True

    # ---------------------------------------------------------------------
    # Pure balanced explore-exploit
    # ---------------------------------------------------------------------
    elif mod.model == 'A4':

        if mod.mode == 'simulation':
            mod.a_init = AbmStructure()
            mod.a_init.agent = 'A4'
            mod.a_init.bayesian_agent = True
            mod.a_init.exploring_agent = True

    # ---------------------------------------------------------------------
    # Pure balanced explore-exploit
    # ---------------------------------------------------------------------
    elif mod.model == 'A5':

        if mod.mode == 'simulation':
            mod.a_init = AbmStructure()
            mod.a_init.agent = 'A5'
            mod.a_init.bayesian_agent = True
            mod.a_init.exploring_agent = True

    # ---------------------------------------------------------------------
    # Pure balanced explore-exploit
    # ---------------------------------------------------------------------
    elif mod.model == 'A6':

        if mod.mode == 'simulation':
            mod.a_init = AbmStructure()
            mod.a_init.agent = 'A6'
            mod.a_init.bayesian_agent = True
            mod.a_init.exploring_agent = True

    # ---------------------------------------------------------------------
    # Pure balanced explore-exploit
    # ---------------------------------------------------------------------
    elif mod.model == 'A7':

        if mod.mode == 'simulation':
            mod.a_init = AbmStructure()
            mod.a_init.agent = 'A7'
            mod.a_init.bayesian_agent = True
            mod.a_init.exploring_agent = True

    # ---------------------------------------------------------------------
    # Pure balanced explore-exploit
    # ---------------------------------------------------------------------
    elif mod.model == 'A8':

        if mod.mode == 'simulation':
            mod.a_init = AbmStructure()
            mod.a_init.agent = 'A7'
            mod.a_init.bayesian_agent = True
            mod.a_init.exploring_agent = True

    return mod
