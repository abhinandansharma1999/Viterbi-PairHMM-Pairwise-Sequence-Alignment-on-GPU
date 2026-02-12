[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zd3UrbhA)
# ECE 213: Assignment 2

## Contents
* [Deadlines](#deadlines)
* [Overview](#overview)
* [Setting up](#setting-up)
* [Code development and testing](#code-development-and-testing)
* [Submission guidelines](#submission-guidelines)

## Deadlines
- Assignment 2: Tuesday, Feb 17 2026 (by 11:59pm PT)

## Overview

The program (`aligner`) reads in a FASTA file containing multiple sequences using the kseq library (http://lh3lh3.users.sourceforge.net/kseq.shtml), transfers them to the GPU, performs a tiling alignment algorithm to compute the traceback path, and then sends the traceback path back to the host to reconstruct the aligned sequences.

In this assignment, students are required to:
1. Parallelize pairwise alignments using one GPU block per alignment pair.
2. Use shared memory to store sequences, and apply memory coalescing for global memory accesses.
3. Apply wavefront parallelism using GPU threads.
4. Parallelize the reconstruction of aligned sequence.

## Setting up

Like before, we will be using UC San Diego's Data Science/Machine Learning Platform ([DSMLP](https://blink.ucsd.edu/faculty/instruction/tech-guide/dsmlp/index.html)) for these assignments.

To get set up with Assignment 2, please follow the steps below:

1. Open and accept the following GitHub Classroom invitation link for assignments 2 through your GitHub account: [https://classroom.github.com/a/zd3UrbhA](https://classroom.github.com/a/zd3UrbhA). A new repository for this will be created specifically for your account (e.g. https://github.com/ECE213-WI26/assgn2-yatisht) and an email will be sent to you via GitHub with the details. 

2. SSH into the DSMLP server (dsmlp-login.ucsd.edu) using the AD account. I recommend using PUTTY SSH client (putty.org) or Windows Subsystem for Linux (WSL) for Windows (https://docs.microsoft.com/en-us/windows/wsl/install-manual). MacOS and Linux users can SSH into the server using the following command (replace `yturakhia` with your username)

```
ssh yturakhia@dsmlp-login.ucsd.edu
```

3. Next, clone the assignment repository in your HOME directory using the following example command (replace repository name `assgn2-yatisht` with the correct name based on step 1) and decompress the data files:
```
cd ~
git clone https://github.com/ECE213-WI26/assgn2-yatisht
cd assgn2-yatisht/data
xz --decompress reference_alignment.fa.xz
xz --decompress sequences.fa.xz
cd ~
```

4. Review the source code (in the `src/` directory). In particular, search `TODO` and `HINT` in `alignment.cu`. Also review the `run-commands.sh` script. This script contains the commands that will be executed via the Docker container on the GPU instance. You may need to modify the commands this script depending on your experiment. Finally, make sure to also review the input test data files in the `data` directory. 
```
cd assgn2-yatisht
```

## Code development and testing

Once your environment is set up on the DSMLP server, you can begin code development and testing using either VS code (that many of you must be familiar with) or if you prefer, using the shell terminal itself (with text editors, such as Vim or Emacs). If you prefer the latter, you can skip the step 1 below.

1. Launch a VS code server from the DSMLP login server using the following command:
   ```
   /opt/launch-sh/bin/launch-codeserver -i ucsdets/datascience-notebook:2022.2-stable
   ```
   If successful, the log of the command will include a message such as:
   ```
   You may access your Code-Server (VS Code) at: http://dsmlp-login.ucsd.edu:14672 using password XXXXXX
   ```
   If the launch command is *unsuccessful*, make sure that there are no already running pods:
   ```
   # View running pods
   kubectl get pods
   # Delete all pods
   kubectl delete pod --all
   ```
   As conveyed in the message of the successful launch command, you can access the VS code server by going to the URL above (http://dsmlp-login.ucsd.edu:14672 in the above example) and entering the password displayed. Note that you may need to use UCSD's VPN service (https://blink.ucsd.edu/technology/network/connections/off-campus/VPN/) if you are performing this step from outside the campus network. Once you gain access to the VS code server from your browser, you can view the directories and files in your DSMLP filesystem and develop code. You can also open a terminal (https://code.visualstudio.com/docs/editor/integrated-terminal) from the VS code interface and run commands on the login server.

2. As mentioned before, we will be using a Docker container, namely `yatisht/ece213-wi26:latest`, for submitting a job on the cluster containing the right virtual environment to build and test the code. This container already contains the correct Cmake version, CUDA and Boost libraries preinstalled within Ubuntu-22.04 OS. Note that these Docker containers use the same filesystem as the DSMLP login server, and hence the files written to or modified by the container is visible to the login server and vice versa. To submit a job that executes `run-commands.sh` script located inside the `assgn2-yatisht` directory on a VM instance with 8 CPU cores, 8 GB RAM and 1 Ampere A30 GPU device (1 GPU is the maximum allowed request on the DSMLP platform), the following command can be executed from the VS Code or DSMLP Shell Terminal (replace the username and directory names below appropriately):

```
ssh yturakhia@dsmlp-login.ucsd.edu /opt/launch-sh/bin/launch.sh -v a30 -c 8 -g 1 -m 8 -i yatisht/ece213-wi26:latest -f ./assgn2-yatisht/run-commands.sh
```
Note that the above command will require you to enter your AD account password again. This command should work and provide a sensible output for the assignment already provided. If you have reached this, you are in good shape to develop and test the code (make sure to modify `run-commands.sh` appropriately before testing). Happy code development!

### Alternative to DSMLP: Google Colab
You can use Google Colab, which provides a cloud-based Jupyter notebook environment with access to GPUs and CPUs. If the DSMLP server is busy or you experience long queue times, you may use Colab as a backup. We **DO NOT** require you to pay for a subscription, and your grade will **not** depend on which platform you use. The choice of platform is entirely based on your personal preference.

1. Claiming your 1-Year Colab Pro Subscription

Google currently offers a free 1-year Colab Pro subscription for students (verified via your .edu email). This provides higher-priority access to GPUs, more RAM, and longer timeout limits.

How to claim: Visit [colab.research.google.com/signup](colab.research.google.com/signup) and follow the student verification steps using your UCSD email.

2. Getting Started with the Provided Notebook

We have provided a [starter Colab notebook](https://colab.research.google.com/drive/1HMjcdZW1lKQLrawVc-gVtwR6PUhpmY8_?usp=sharing) for assignments. Open the link and **click File > Save a copy in Drive**. This creates a personal version in your own Google Drive that you can edit and run. 

## Submission guidelines

* Make sure to keep your code repository (e.g. https://github.com/ECE213-WI26/assgn2-yatisht) up-to-date.
* All new files (such as figures and reports) should be uploaded to the `submission-files` directory in the repository
* Once you are ready to submit, create a [new release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository#creating-a-release) for your submission with a tag name `v1.0`.
* Provide a good description for the changes you have made and any information that you would like to be conveyed during the grading.
* Submit the URL corresponding to the release to Canvas for your assignment submission (e.g. https://github.com/ECE213-WI26/assgn2-yatisht/releases/tag/v1.0; note that you are only required to submit a URL via Canvas). Only releases will be considered for grading and the date and time of the submitted release will be considered final for the late day policy. Be mindful of the deadlines.