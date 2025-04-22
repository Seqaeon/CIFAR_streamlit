import os
import numpy as np
import streamlit as st
#from streamlit_drawable_canvas import st_canvas
import streamlit_analytics2
from PIL import Image

import data_prep as data

import ao_core as ao
from arch__CIFAR import arch


def streamlit_setup():
    if "agent" not in st.session_state:
        st.session_state.agent = setup_agent()
    return


def setup_agent():
    agent = ao.Agent(arch, save_meta=False, uid = "cifar",notes = "Default Agent")
    return agent


def initialize_session_state():
    if "interrupt" not in st.session_state:
        st.session_state.interrupt = False


def reset_interrupt():
    st.session_state.interrupt = False


def set_interrupt():
    st.session_state.interrupt = True


def run_agent(user_STEPS, INPUT, LABEL=[]):
    # running the Agent
    st.session_state.agent.reset_state()
    print(LABEL)
    if np.shape(LABEL)[0] == 0:
        for x in np.arange(user_STEPS):
            print("step: " + str(x))
            # core method to run Agents
            st.session_state.agent.next_state(INPUT, DD=False, unsequenced=True)

    # Config 1 -->         
    else:
        print("labelled")
        # core method to run Agents
        st.session_state.agent.next_state(INPUT, LABEL, DD=False, unsequenced=True)

    # saving results
    s = st.session_state.agent.state
    q_index = st.session_state.agent.arch.Q__flat
    # z = st.session_state.agent.arch.Z__flat
    z_index = st.session_state.agent.arch.Z__flat
    st.session_state.agent_qresponse = np.reshape(
        st.session_state.agent.story[s - 1, q_index], [32, 32,3,8]
    )
    # st.session_state.agent_zresponse = st.session_state.agent.story[s, z]
    z = st.session_state.agent.story[s - 1, z_index]

    # return st.session_state.agent_zresponse
    return z

# Initialize selected_z only if it doesn't exist

if 'selected_z' not in st.session_state:
    st.session_state.selected_z = {}
if 'cifar10_labels' not in st.session_state:
    _,_,labels_10 =  data.process_data(dataset='CIFAR-10')
    st.session_state.cifar10_labels = labels_10
if 'cifar100_labels' not in st.session_state:
    _,_,labels_100 =  data.process_data(dataset='CIFAR-100')
    st.session_state.cifar100_labels = labels_100
if 'chosen_set' not in st.session_state:
    st.session_state.chosen_set = 'CIFAR-10'



def run_trials(is_training, num_trials, user_STEPS):
    #st.session_state.selected_z = []


    initialize_session_state()
    
    st.session_state.is_training = is_training
    
    chosen_set = st.session_state.training_sets
    st.session_state.chosen_set = chosen_set

    (MN_TRAIN, MN_TRAIN_Z), (MN_TEST, MN_TEST_Z), label_names = data.process_data(chosen_set)




    trialset = MN_TRAIN if is_training else MN_TEST
    trialset_z = MN_TRAIN_Z if is_training else MN_TEST_Z

    # selected_in, selected_z = data.random_sample_same(num_trials, trialset, trialset_z, seed=42)
    selected_in, selected_z = data.random_sample(num_trials, trialset, trialset_z)



    correct_responses = 0
    num_trials = len(selected_in)
#    repeats = 2 if is_training else 11
#    st.session_state.selected_z.extend([num for num in selected_z for _ in range(repeats)])
    if is_training:
        INPUT = data.bitmap_to_binary(selected_in).reshape(num_trials, 32*32*3*8)
        st.session_state.agent.next_state_batch(INPUT, selected_z, unsequenced=True)
        repeats = 2
        selected_z_i = [num for num in selected_z for _ in range(repeats)]
        len_z = len(st.session_state.selected_z.keys())
        k = {i: [label_names[bin_to_pix(selected_z_i[i-(len_z+1)]).item()], chosen_set, True] for i in range(len_z+1, len_z+1+len(selected_z_i))}
        st.session_state.selected_z.update(k)
        #st.session_state.selected_z.extend([num for num in selected_z for _ in range(repeats)])
        print("Training complete; neurons updated.")
        return

    st.session_state.num_trials_actual = 0

    progress_bar = st.progress(float(0))
    for t in np.arange(num_trials):
        nt = t/num_trials
        progress_bar.progress( nt, text="Testing in Progress")

        @st.dialog("Process Interrupted")
        def interrupt_modal_dialog():
            st.warning(
                "Function interrupted! Click the *Re-Enable Processing* button in the sidebar to train/test again."
            )

        if st.session_state.interrupt:
            interrupt_modal_dialog()
            break

        INPUT = data.bitmap_to_binary(selected_in[t, :, :]).reshape(32*32*3*8)
        LABEL = selected_z[t]
        
        if is_training:
            user_STEPS = 1
            run_agent(user_STEPS, INPUT, LABEL)
            print("Trained on " + str(t))
#            repeats = 2
#            st.session_state.selected_z.extend([LABEL]*repeats)
        else:
            response_agent = run_agent(user_STEPS, INPUT, LABEL=[])
            if np.array_equal(response_agent, LABEL):
                correct_responses += 1
            print("Tested on " + str(t))
            print("TOTAL CORRECT-----------------" + str(correct_responses))
            repeats = user_STEPS + 1
            LABEL_i = [LABEL]*repeats
            len_l = len(st.session_state.selected_z.keys())
            k = {i: [label_names[bin_to_pix(LABEL_i[i-(len_l+1)]).item()], chosen_set, False] for i in range(len_l+1, len_l+1+len(LABEL_i))}
            st.session_state.selected_z.update(k)
            #st.session_state.selected_z.extend([LABEL]*repeats)

        st.session_state.num_trials_actual += 1

        trial_result = (correct_responses / st.session_state.num_trials_actual) * 100
        st.session_state.correct_responses = correct_responses
        st.session_state.trial_result = trial_result
        print("Correct on {x}%".format(x=trial_result))
        # return correct_responses
    progress_bar.empty()


def run_uploaded_image():
    input = data.bitmap_to_binary(st.session_state.uploaded_image).reshape(32*32*3*8)
    label = []
    user_steps = 10
    label_i = st.session_state.cifar10_labels.index(st.session_state.uploaded_label) if st.session_state.chosen_set == 'CIFAR-10' else st.session_state.cifar100_labels.index(st.session_state.uploaded_label)
    if st.session_state.train_uploaded:

        label = list(np.binary_repr(int(label_i), 7))
        user_steps = 1
    repeats = user_steps + 1
    selected_z_i = [st.session_state.uploaded_label for _ in range(repeats)]
    #print(selected_z_i)
    len_z = len(st.session_state.selected_z.keys())
    #print(len_z)
    k = {i: [selected_z_i[i-(len_z+1)], st.session_state.chosen_set, True if st.session_state.train_uploaded else False] for i in range(len_z+1, len_z+1+len(selected_z_i))}
    st.session_state.selected_z.update(k)
    response = run_agent(user_steps, input, LABEL=label)
    print(response)
    response_int = int("".join(str(x) for x in response), 2)
    st.session_state.image_int = response_int
    return


# Used to construct images of agent state
def bin_to_pix(img):
    if img.ndim == 1:
        img_int = [int(i) for i in img]
        return np.array(int(''.join(map(str, img_int)), 2),dtype=np.uint8)
    # elif img.ndim == 2:
    #     return np.array([int(''.join(map(str, pixel)), 2) for pixel in img], dtype=np.uint8)
    else:
        return np.array([bin_to_pix(sub_array) for sub_array in img],dtype=np.uint8)




def arr_to_img(img_array, enlarge_factor=15):
    """
    Convert an array to an RGB image and enlarge it.

    Args:
        img_array (np.ndarray): Input array (can be 1D, 2D, or 3D).
        enlarge_factor (int): Factor to enlarge the image.

    Returns:
        PIL.Image: Enlarged RGB image.
    """

    if img_array.ndim == 1:
        img_array = np.array(img_array, dtype=np.uint8)
        img_array = img_array * 255

    elif img_array.ndim == 2:
        img_array = bin_to_pix(img_array)

        img_array = np.stack([img_array] * 3, axis=-1)

    elif img_array.ndim == 3:


        if img_array.shape[2] != 3:
            raise ValueError("Input array must have 3 channels for RGB images.")
    elif img_array.ndim == 4:
        img_array = bin_to_pix(img_array)

        if img_array.shape[2] != 3:
            raise ValueError("Input array must have 3 channels for RGB images.")

    else:
        raise ValueError("Input array must be 1D, 2D, or 3D.")

    # Enlarge the image
    enlarged_array = np.repeat(img_array, enlarge_factor, axis=0)

    try:
        enlarged_array = np.repeat(enlarged_array, enlarge_factor, axis=1)
        img = Image.fromarray(enlarged_array, mode="RGB")
    except:
        enlarged_array = np.tile(enlarged_array, [enlarge_factor, 1])
        img = Image.fromarray(enlarged_array, mode="L")
        pass

    return img
streamlit_analytics2.start_tracking()
# Basic streamlit setup
st.set_page_config(
    page_title="CIFAR Demo by AO Labs",
    page_icon="misc/ao_favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://discord.gg/Zg9bHPYss5",
        "Report a bug": "mailto:eng@aolabs.ai",
        "About": "AO Labs builds next-gen AI models that learn after training; learn more at docs.aolabs.ai/docs/mnist-benchmark",
    },
)

streamlit_setup()

st.title("Understanding *Weightless* NNs via CIFAR")
st.write("### *a demo by [aolabs.ai](https://www.aolabs.ai/)*")

train_max = 60000
test_max = 10000

############################################################################
with st.sidebar:
    st.write("## Current Active Agent:")
    st.write(st.session_state.agent.notes)

    start_button = st.button(
        "Re-Enable Training & Testing",
        on_click=reset_interrupt,
        help="If you stopped a process\n click to re-enable Testing/Training agents.",
    )
    stop_button = st.button(
        "Stop Testing",
        on_click=set_interrupt,
        help="Click to stop a current Test if it is taking too long.",
    )

    st.write("---")
    st.write("## Load Agent:")

    def load_pickle_files(directory):
        pickle_files = [
            f[:-10] for f in os.listdir(directory) if f.endswith(".ao.pickle")
        ]  # [:-10] is to remove the "ao.pickle" file extension
        return pickle_files

    # directory_option = st.radio(
    #     "Choose directory to retrieve Agents:",
    #     ("App working directory", "Custom directory"),
    #     label_visibility="collapsed"
    # )
    # if directory_option == "App working directory":
    directory = os.path.dirname(os.path.abspath(__file__))
    # else:
    #     directory = st.text_input("Enter a custom directory path:")

    if directory:
        pickle_files = load_pickle_files(directory)

        if pickle_files:
            selected_file = st.selectbox(
                "Choose from saved Agents:", options=pickle_files
            )

            if st.button(f"Load {selected_file}"):
                file_path = os.path.join(directory, selected_file)
                st.session_state.agent = ao.Agent.unpickle(
                    file=file_path, custom_name=selected_file
                )
                st.session_state.agent._update_neuron_data()
                st.write("Agent loaded")
        else:
            st.warning("No Agents saved yet-- be the first!")

    st.write("---")
    st.write("## Save Agent:")

    agent_name = st.text_input(
        "## *Optional* Rename active Agent:", value=st.session_state.agent.notes
    )
    st.session_state.agent.notes = agent_name

    @st.dialog("Save successful!")
    def save_modal_dialog():
        st.write("Agent saved to your local disk (in the same directory as this app).")

    agent_name = agent_name.split("\\")[-1].split(".")[0]
    if st.button("Save " + agent_name):
        st.session_state.agent.pickle(agent_name)
        save_modal_dialog()

    st.write("---")
    st.write("## Download/Upload Agents:")

    @st.dialog("Upload successful!")
    def upload_modal_dialog():
        st.write(
            "Agent uploaded and ready as *Newly Uploaded Agent*, which you can rename during saving."
        )

    uploaded_file = st.file_uploader(
        "Upload .ao.pickle files here", label_visibility="collapsed"
    )
    if uploaded_file is not None:
        if st.button("Confirm Agent Upload"):
            st.session_state.agent = ao.Agent.unpickle(
                uploaded_file, custom_name="Newly Uploaded Agent", upload=True
            )
            st.session_state.agent._update_neuron_data()
            upload_modal_dialog()

    @st.dialog("Download ready")
    def download_modal_dialog(agent_pickle):
        st.write(
            "The Agent's .ao.pickle file will be saved to your default Downloads folder."
        )

        # Create a download button
        st.download_button(
            label="Download Agent: " + st.session_state.agent.notes,
            data=agent_pickle,
            file_name=st.session_state.agent.notes,
            mime="application/octet-stream",
        )

    if st.button("Prepare Active Agent for Download"):
        agent_pickle = st.session_state.agent.pickle(download=True)
        download_modal_dialog(agent_pickle)
############################################################################

agent_col, state_col = st.columns(2)

with agent_col:
    with st.expander("#### Batch Training & Testing", expanded=True):
        st.write("---")
        st.write("##### Training")

        training_set_options = data.dataset_list 
        st.session_state.training_sets = st.selectbox(
            "Select training datasets:",
            options=training_set_options,
            index=training_set_options.index("CIFAR-10"),
            help="When training on standard fonts (eg. Times New Roman, Arial, etc.), it trains on all of the digits of that font.",
        )

        
        
        train_count = st.number_input(
            "Set the number of training pairs:",
            1,
            train_max,
            value=2,
            help="Randomly  selected from CIFAR's training set.",
        )
        
        st.button(
            "Train Agent",
            on_click=run_trials,
            args=(True, train_count, 1),
            disabled=len(st.session_state.chosen_set) == 0,
        )


        st.write("---")
        st.write("##### Testing")
        t_count, t_steps = st.columns(2)
        with t_count:
            test_count = st.number_input(
                "Number of test images",
                1,
                test_max,
                value=1,
                help="Randomly selected from CIFAR's 10k test set.",
            )
        with t_steps:
            user_STEPS = st.number_input(
                "Number of steps per test image:",
                1,
                20,
                value=10,
                help="10 is a good default; this level of agent usually converges on a stable pattern after ~7 steps (if you've trained it enough).",
            )
        st.button(
            "Test Agent", on_click=run_trials, args=(False, test_count, user_STEPS)
        )

        st.write("---")

        # display trial result
        if "trial_result" in st.session_state:
            st.write("##### Test Results")
            st.write(
                "The agent predicted {correct} out of {total} images correctly, an accuracy of:".format(
                    correct=st.session_state.correct_responses,
                    total=st.session_state.num_trials_actual,
                )
            )
            st.write("# {result}%".format(result=st.session_state.trial_result))

    with st.expander("#### Continuous Learning", expanded=True):
        st.write(
            f"You can also train or test your agent on custom inputs: try uploading an image of one of the object classes here:{st.session_state.cifar10_labels if st.session_state.chosen_set == 'CIFAR-10' else st.session_state.cifar100_labels}"
        )

        t_upload, t_label = st.columns(2)
        with t_upload:
            # File uploader
            uploaded_file = st.file_uploader(
                "Upload an image", 
                type=["png", "jpg", "jpeg"],  # Allowed file types
                key="file_uploader"
            )

            # Display the uploaded image
            if uploaded_file is not None:
                # Open the image using PIL
                uploaded_image = Image.open(uploaded_file)
                
                # Display the image
                st.image(uploaded_image.resize((320,320)), caption="Uploaded Image", use_container_width=True)

        with t_label:
            st.session_state.train_uploaded = st.toggle("Train on Uploaded Image")
            st.session_state.uploaded_label = st.selectbox(
                "Select Label for Uploaded image:",
                options= st.session_state.cifar10_labels if st.session_state.chosen_set == 'CIFAR-10' else st.session_state.cifar100_labels,
                index=0,
                #help="When training on standard fonts (eg. Times New Roman, Arial, etc.), it trains on all of the digits of that font.",
            )

        if uploaded_file is not None:
            input_numpy_array = np.array(uploaded_image)
            input_image = Image.fromarray(input_numpy_array.astype("uint8"), "RGB")
            input_image_gs = input_image.convert("RGB")
            resized_gs = input_image_gs.resize((32, 32), Image.Resampling.LANCZOS)
            np_gs = np.array(resized_gs)
            st.session_state.uploaded_image = np_gs

        if st.session_state.train_uploaded:
            Image_button_text = "Train on Uploaded image with label: " + str(
                st.session_state.uploaded_label
            )
        else:
            Image_button_text = "Test on Uploaded image"

        st.button(Image_button_text, on_click=run_uploaded_image)

        if "image_int" in st.session_state:
            st.write("Image identified as:")
            st.write("# {x}".format(x=st.session_state.cifar10_labels[st.session_state.image_int] if st.session_state.chosen_set == 'CIFAR-10' else st.session_state.cifar100_labels[st.session_state.image_int]))

with state_col:
    st.write("#### Agent Visual Inspector - view the agent's state history")
    instruction_md = """
    Weightless neural network agents function as *neural state machines*, so during Testing, an agent is shown an image from MNIST and its inner and output states will change in response, allowing you to 'see' what the agent is thinking (unlike deep learning which remains a blackbox); the final output state is translated into an integer label to determine the accuracy of the agent's inference. \n
    You can view all that information by cycle through the states below. \n
    * ***Input*** is the 28x28 B&W pixel input to the agent from MNIST or your canvas (MNIST is grayscale but for this demo we're downsampling to B&W). \n
    * ***Inner State*** is visual representation of 28x28 binary neurons that make up the agent's inner or hidden layer (the same shape as the input, to aid with visual inspection.) \n
    * ***Output State*** is a visual representation of 4 binary neurons (also displayed as a list) that make up the agent's output layer (the states of the binary neurons are translated to an integer label, 0-9). \n
    \n
    Starting from state 1, you'll first cycle through the training data you fed the agent-- you'll notice there's noise interspersed between the training states; this is because we're not tasking the agent with learning a sequence between the MNIST data, so we introduce randomness in between. \n
    When you cycle through to the testing states, you'll see a fixed input with an evolving inner and output states. Often they'll converge on a pattern which correlates with the label of the input image. \n
    """
    with st.expander("About"):
        st.markdown(instruction_md)

    if st.session_state.agent.state - 1 == 0:
        min_value = 0
    else:
        min_value = 1

    sel_state = st.number_input(
        "Displaying state:",
        min_value,
        st.session_state.agent.state,
        value=st.session_state.agent.state - 1,
        help="The agent has history up until state: {}".format(
            st.session_state.agent.state
        ),
    )

    I_col, Q_col, Z_col = st.columns(3)
#    if st.session_state.is_training:
#        sel_state_i = sel_state #- 1
#    else:
#        sel_state_i = sel_state

    with I_col:
        st.write("##### Input")

        i_arr = st.session_state.agent.story[
            sel_state, st.session_state.agent.arch.I__flat
        ]
        i_arr = np.reshape(i_arr, [32, 32,3,8])
        i_img = arr_to_img(i_arr)
        st.image(i_img)
        #st.write(sel_state)

    with Q_col:
        st.write("##### Inner State")
        q_arr = st.session_state.agent.story[
            sel_state, st.session_state.agent.arch.Q__flat
        ]
        q_arr = np.reshape(q_arr, [32, 32,3,8])
        q_img = arr_to_img(q_arr)
        st.image(q_img)

    with Z_col:
        st.write("##### Output State")
        z_arr = st.session_state.agent.story[
            sel_state, st.session_state.agent.arch.Z__flat
        ]
        z_int = z_arr.dot(2 ** np.arange(z_arr.size)[::-1])


        z_img = arr_to_img(z_arr)
        st.write("Result in binary:")

        st.image(z_img)
        st.write("  " + str(z_arr))

#        print(sel_state)
#
#        print(len(st.session_state.selected_z.keys()))



        

        if st.session_state.selected_z and sel_state <= len(st.session_state.selected_z.keys()):
            print("First")

            if not st.session_state.selected_z[sel_state][2]:
                st.write("Actual label: "+ st.session_state.selected_z[sel_state][0])
                if st.session_state.selected_z[sel_state][1] == 'CIFAR-10':

                    st.write("Result as an label name: " + st.session_state.cifar10_labels[z_int] if z_int <= 9 else '')
                elif st.session_state.selected_z[sel_state][1] == 'CIFAR-100':
                    st.write("Result as an label name: " + st.session_state.cifar100_labels[z_int] if z_int <= 99 else '')

                else:

                    st.write("Result as an label name: " + '')

            else:
                if st.session_state.selected_z[sel_state][1] == 'CIFAR-10':
                    st.write("Label name: " + st.session_state.cifar10_labels[z_int] if z_int <= 9 else '')
                elif st.session_state.selected_z[sel_state][1] == 'CIFAR-100':
                    st.write("Label name: " + st.session_state.cifar100_labels[z_int] if z_int <= 99 else '')
                else:
                    st.write("Label name: " + '')


#

st.write("---")
footer_md = """
    [View & fork the code behind this application here.](https://github.com/aolabsai/MNIST_streamlit) \n
    To learn more about Weightless Neural Networks and the new generation of AI we're developing at AO Labs, [visit our docs.aolabs.ai.](https://docs.aolabs.ai/docs/mnist-benchmark)\n
    \n
    We eagerly welcome contributors and hackers at all levels! [Say hi on our discord.](https://discord.gg/Zg9bHPYss5)
    """
st.markdown(footer_md)
st.image("misc/aolabs-logo-horizontal-full-color-white-text.png", width=300)

streamlit_analytics2.stop_tracking()
