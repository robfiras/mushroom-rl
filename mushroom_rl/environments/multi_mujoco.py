import mujoco
import numpy as np
from mushroom_rl.environments import MuJoCo
from mushroom_rl.core import MDPInfo
from .mult_MujocoGlfwViewer import MultMujocoGlfwViewer
from mushroom_rl.utils.spaces import Box
from mushroom_rl.utils.mujoco import *


class MultiMuJoCo(MuJoCo):
    """
    Class to create N Mushroom environments at the same time using the MuJoCo simulator. This class is not meant to run
    N environments in parallel, but to load N environments and randomly sample one of the environment every episode.

    """

    def __init__(self, file_names, actuation_spec, observation_spec, gamma, horizon, timestep=None,
                 n_substeps=1, n_intermediate_steps=1, additional_data_spec=None, collision_groups=None, **viewer_params):
        """
        Constructor.

        Args:
             file_names (list): The list of paths to the XML files to create the
                environments;
             actuation_spec (list): A list specifying the names of the joints
                which should be controllable by the agent. Can be left empty
                when all actuators should be used;
             observation_spec (list): A list containing the names of data that
                should be made available to the agent as an observation and
                their type (ObservationType). They are combined with a key,
                which is used to access the data. An entry in the list
                is given by: (key, name, type);
             gamma (float): The discounting factor of the environment;
             horizon (int): The maximum horizon for the environment;
             timestep (float): The timestep used by the MuJoCo
                simulator. If None, the default timestep specified in the XML will be used;
             n_substeps (int, 1): The number of substeps to use by the MuJoCo
                simulator. An action given by the agent will be applied for
                n_substeps before the agent receives the next observation and
                can act accordingly;
             n_intermediate_steps (int, 1): The number of steps between every action
                taken by the agent. Similar to n_substeps but allows the user
                to modify, control and access intermediate states.
             additional_data_spec (list, None): A list containing the data fields of
                interest, which should be read from or written to during
                simulation. The entries are given as the following tuples:
                (key, name, type) key is a string for later referencing in the
                "read_data" and "write_data" methods. The name is the name of
                the object in the XML specification and the type is the
                ObservationType;
             collision_groups (list, None): A list containing groups of geoms for
                which collisions should be checked during simulation via
                ``check_collision``. The entries are given as:
                ``(key, geom_names)``, where key is a string for later
                referencing in the "check_collision" method, and geom_names is
                a list of geom names in the XML specification.
             **viewer_params: other parameters to be passed to the viewer.
                See MujocoGlfwViewer documentation for the available options.

        """
        # Create the simulation
        assert type(file_names)
        self._models = [mujoco.MjModel.from_xml_path(f) for f in file_names]
        self._model = self._models[0]
        if timestep is not None:
            self._model.opt.timestep = timestep
            self._timestep = timestep
        else:
            self._timestep = self._model.opt.timestep

        self._datas = [mujoco.MjData(m) for m in self._models]
        self._data = self._datas[0]

        self._n_intermediate_steps = n_intermediate_steps
        self._n_substeps = n_substeps
        self._viewer_params = viewer_params
        self._viewer = None
        self._obs = None

        # Read the actuation spec and build the mapping between actions and ids
        # as well as their limits
        if len(actuation_spec) == 0:
            self._action_indices = [i for i in range(0, len(self._data.actuator_force))]
        else:
            self._action_indices = []
            for name in actuation_spec:
                self._action_indices.append(self._model.actuator(name).id)

        low = []
        high = []
        for index in self._action_indices:
            if self._model.actuator_ctrllimited[index]:
                low.append(self._model.actuator_ctrlrange[index][0])
                high.append(self._model.actuator_ctrlrange[index][1])
            else:
                low.append(-np.inf)
                high.append(np.inf)
        action_space = Box(np.array(low), np.array(high))

        # Read the observation spec to build a mapping at every step. It is
        # ensured that the values appear in the order they are specified.
        self.obs_helpers = [ObservationHelper(observation_spec, m, d, max_joint_velocity=3)
                            for m, d in zip(self._models, self._datas)]
        self.obs_helper = self.obs_helpers[0]

        observation_space = Box(*self.obs_helper.get_obs_limits())

        # multi envs with different obs limits are now allowed, do sanity check
        for oh in self.obs_helpers:
            low, high = self.obs_helper.get_obs_limits()
            if  not np.array_equal(low, observation_space.low) or not np.array_equal(high, observation_space.high):
                raise ValueError("The provided environments differ in the their observation limits. "
                                 "This is not allowed.")

        # Pre-process the additional data to allow easier writing and reading
        # to and from arrays in MuJoCo
        self.additional_data = {}
        if additional_data_spec is not None:
            for key, name, ot in additional_data_spec:
                self.additional_data[key] = (name, ot)

        # Pre-process the collision groups for "fast" detection of contacts
        self.collision_groups = {}
        if collision_groups is not None:
            for name, geom_names in collision_groups:
                self.collision_groups[name] = {mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                                               for geom_name in geom_names}

        # Finally, we create the MDP information and call the constructor of
        # the parent class
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        mdp_info = self._modify_mdp_info(mdp_info)

        # set the warning callback to stop the simulation when a mujoco warning occurs
        mujoco.set_mju_user_warning(self.user_warning_raise_exception)

        # call grad-parent class, not MuJoCo
        super(MuJoCo, self).__init__(mdp_info)


    def render(self):
        if self._viewer is None:
            self._viewer = MultMujocoGlfwViewer(self._model, self.dt, **self._viewer_params)

        self._viewer.render(self._data)

    def reset(self, obs=None):
        mujoco.mj_resetData(self._model, self._data)
        self.setup()

        i = np.random.randint(0, len(self._models))
        self._model = self._models[i]
        self._data = self._datas[i]
        self.obs_helper = self.obs_helpers[i]
        mujoco.mj_resetData(self._model, self._data)

        if self._viewer is not None:
            self._viewer.load_new_model(self._model)

        self._obs = self._create_observation(self.obs_helper.build_obs(self._data))
        return self._modify_observation(self._obs)

