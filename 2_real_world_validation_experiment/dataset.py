"""
Copyright (c) 2024 Harvard Ability lab
Title: "A smartphone activity monitor that accurately estimates energy expenditure"
"""

import os
import numpy as np
import pandas as pd
import pickle
import warnings
from scipy import signal
from numpy import linalg as LA  
import matplotlib.pyplot as plt  

# Ignore specific warning
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class rwDataset:
    """
    Class for processing and analyzing real-world gait data.
    """

    def __init__(self, params):
        self.params = params
        self.target_subj = self.params['target_subj']  # Target subject identifier
        self.val_data_path = self.params['val_data_path']
        self.preprocessing()

    def get_cum_met(self, met, time):
        """
        Calculate cumulative metabolic cost.

        Args:
            met (numpy array): Array of metabolic values.
            time (numpy array): Array of time values.

        Returns:
            float: Cumulative metabolic cost.
        """
        met_out = sum(met[idx + 1] * dt for idx, dt in enumerate(np.diff(time)))
        return met_out

    def checkPeaks(self, strike_vec, peak_height_thresh, peak_min_dist):
        """
        Detect peaks in the strike vector.

        Args:
            strike_vec (numpy array): Input vector to find peaks.
            peak_height_thresh (float): Minimum height of peaks.
            peak_min_dist (int): Minimum distance between peaks.

        Returns:
            numpy array: Indices of detected peaks.
        """
        peak_list = signal.find_peaks(strike_vec, height=peak_height_thresh, distance=peak_min_dist)
        return peak_list[0]

    def processRawGait(self, data_array, start_ind, end_ind, age, gen, weight, height, num_bins=30):
        """
        Process raw gait data for feature extraction.

        Args:
            data_array (numpy array): Array of raw gait data.
            start_ind (int): Start index of the gait cycle.
            end_ind (int): End index of the gait cycle.
            age (float): Age of the subject.
            gen (str): Gender of the subject.
            weight (float): Weight of the subject.
            height (float): Height of the subject.
            num_bins (int): Number of bins for discretization.

        Returns:
            numpy array: Flattened array of extracted features.
        """
        # Crop to the gait cycle
        gait_data = data_array[start_ind:end_ind, :]
        
        # Duration of stride in seconds
        dur_stride = gait_data.shape[0] / 100
        
        # Resample gait cycle data into a fixed number of bins
        bin_gait = signal.resample(gait_data, num_bins, axis=0)
        
        # Transpose to shape [features x bins] for flattening
        shift_flip_bin_gait = bin_gait.T
        model_input = shift_flip_bin_gait.flatten()[:90]  # Flatten and take the first 90 elements

        # Split into gyro x, y, and z components
        gyro_x = model_input[:30]
        gyro_y = model_input[30:60]
        gyro_z = model_input[60:90]

        def get_features(signal):
            """
            Extract statistical features from the signal.

            Args:
                signal (numpy array): Input signal.

            Returns:
                numpy array: Array of extracted features.
            """
            from scipy.stats import skew
            mean_ = np.mean(signal)
            std_ = np.std(signal)
            med_ = np.median(signal)
            skew_ = skew(signal)
            pow_ = LA.norm(signal, ord=2)  # Power as the Euclidean norm
            features = np.array([mean_, std_, med_, skew_, pow_])
            return features

        # Extract features for each gyro component
        gyro_x_feat = get_features(gyro_x)
        gyro_y_feat = get_features(gyro_y)
        gyro_z_feat = get_features(gyro_z)

        # Concatenate all features into a single array
        model_input = np.concatenate((
            model_input.reshape(-1, 1), 
            gyro_x_feat.reshape(-1, 1),
            gyro_y_feat.reshape(-1, 1),
            gyro_z_feat.reshape(-1, 1)
        ), axis=0).flatten()
        
        # Add weight, height, and stride duration as additional features
        model_input = np.insert(model_input, 0, [weight, height, dur_stride])

        return model_input

    def estimateHeartrate(self, hr_val, weight, age, gender):
        """
        Estimate energy expenditure based on heart rate.

        Args:
            hr_val (float): Heart rate value.
            weight (float): Weight of the subject.
            age (float): Age of the subject.
            gender (str): Gender of the subject.

        Returns:
            float: Estimated energy expenditure in watts.
        """
        # Encode gender as 1 for male, 0 for female
        gender_code = 1 if gender == 'M' else 0
        
        # Calculate energy expenditure using gender-specific equations
        ee_est = (
            gender_code * (-55.0969 + 0.6309 * hr_val + 0.1988 * weight + 0.2017 * age) 
            + (1 - gender_code) * (-20.4022 + 0.4472 * hr_val - 0.1263 * weight + 0.074 * age)
        )
        
        # Convert kJ/min to watts
        ee_est *= 16.6666666667
        return ee_est

    def estimatePedometer(self, abs_time, peak_index_list, stride_detect_window, gender, weight):
        """
        Estimate energy expenditure using a pedometer.

        Args:
            abs_time (numpy array): Array of absolute time values.
            peak_index_list (list): List of peak indices indicating gait cycles.
            stride_detect_window (int): Maximum allowed stride duration in samples.
            gender (str): Gender of the subject.
            weight (float): Weight of the subject.

        Returns:
            tuple: Arrays of time and estimated energy expenditure.
        """
        # Encode gender as 1 for male, 0 for female
        gender_code = 1 if gender == 'M' else 0

        ee_ped = []
        ee_ped_time = []

        for i in range(len(peak_index_list) - 1):
            gait_start_index = peak_index_list[i]
            gait_stop_index = peak_index_list[i + 1]
            time_s = abs_time[gait_start_index]
            time_e = abs_time[gait_stop_index]

            if (gait_stop_index - gait_start_index) <= stride_detect_window:
                cad = 2 / (time_e - time_s) * 60  # Calculate cadence in steps/min

                # Calculate MET and energy expenditure based on gender
                MET = -7.065 + (0.105 * cad) if gender_code == 1 else -8.805 + (0.110 * cad)
                ee_est = MET / 1.162 * weight
                ee_est = max(ee_est, 0)  # Ensure energy expenditure is non-negative
                ee_ped.append(ee_est)
                ee_ped_time.append(time_s)

        num_points = int((gait_stop_index - gait_start_index) / 100)
        static_idx = np.linspace(gait_start_index, gait_stop_index, num=num_points).astype(int)[1:-1]
        for cur_idx in static_idx:
            ee_ped.append(ee_est)
            ee_ped_time.append(abs_time[cur_idx])

        return np.array(ee_ped_time), np.array(ee_ped)

    def estimateMetabolics(self, abs_time, est_model, gait_data, x_data, peak_index_list, stride_detect_window, num_bins=30):
        """
        Estimate metabolic cost from gait data.

        Args:
            abs_time (numpy array): Array of absolute time values.
            est_model (model): Pre-trained estimation model.
            gait_data (numpy array): Array to store gait data.
            x_data (numpy array): Input data array.
            peak_index_list (list): List of peak indices indicating gait cycles.
            stride_detect_window (int): Maximum allowed stride duration in samples.
            num_bins (int): Number of bins for resampling.

        Returns:
            tuple: Arrays of time, absolute time, estimated energy expenditure, and processed gait data.
        """
        gait_cnt = 0
        ee_est = []
        ee_time = []
        ee_abs_time = []

        for i in range(len(peak_index_list) - 1):
            gait_start_index = peak_index_list[i]
            gait_stop_index = peak_index_list[i + 1]

            if (gait_stop_index - gait_start_index) <= stride_detect_window:
                cur_gait_data = self.processRawGait(
                    x_data, gait_start_index, gait_stop_index, self.age, 
                    self.gender, self.body_weight, self.height, num_bins
                )

                gait_data[:, gait_cnt] = cur_gait_data

                model_input = cur_gait_data.reshape(1, -1)
                cur_ee_est = est_model.predict(model_input)[0]

                ee_est.append(cur_ee_est)
                ee_time.append(gait_start_index)
                ee_abs_time.append(abs_time[gait_start_index])
                gait_cnt += 1
            else:
                num_points = int((gait_stop_index - gait_start_index) / 100)
                static_idx = np.linspace(gait_start_index, gait_stop_index, num=num_points).astype(int)[1:-1]
                for cur_idx in static_idx:
                    ee_est.append(self.basal_rate)
                    ee_time.append(cur_idx)
                    ee_abs_time.append(abs_time[cur_idx])

        return np.array(ee_time), np.array(ee_abs_time), np.array(ee_est), np.array(gait_data)

    def preprocessing(self):
        """
        Preprocess data and calculate absolute error for various metrics.
        """
        abs_err_openmet_cum_list = []
        abs_err_sw_cum_list = []
        abs_err_hr_cum_list = []
        abs_err_ped_cum_list = []

        for cur_subj in self.target_subj:
            print(cur_subj)
            val_data_path = self.val_data_path

            # Load respirometry data
            csv_file_path = os.path.join(val_data_path, cur_subj, 'respirometry_met.csv')
            df_respirometry = pd.read_csv(csv_file_path)
            cur_gt_met_time = df_respirometry['time (s)'].values
            cur_gt_met = df_respirometry['metabolics (W)'].values

            # Load thigh IMU data
            csv_file_path = os.path.join(val_data_path, cur_subj, 'imu_thigh.csv')
            df_imu_data = pd.read_csv(csv_file_path)

            # Load subject-specific information
            csv_file_path = os.path.join(val_data_path, cur_subj, 'subject_spec_info.csv')
            df_subj_info = pd.read_csv(csv_file_path)

            self.age = df_subj_info['age (y)'].values
            self.gender = df_subj_info['gender'].values
            self.body_weight = df_subj_info['weight (kg)'].values
            self.height = df_subj_info['height (m)'].values
            self.basal_rate = df_subj_info['basal rate (W)'].values
            self.rest_met = df_subj_info['rest metabolics (W)'].values

            num_bins = self.params['num_bins']
            fs = 100
            peak_height_thresh = 70
            peak_min_dist = int(0.6 * fs)
            stride_detect_window = 4 * fs

            # Load IMU data
            imu_time = df_imu_data['time (s)'].values
            imu_gyro_z = df_imu_data['gyro_z (rad/s)'].values
            imu_gyro_acc = df_imu_data.values[:, 1:]

            # Peak detection for gait segmentation
            thigh_peak = self.checkPeaks(np.rad2deg(imu_gyro_z), peak_height_thresh, peak_min_dist)
            gait_data_thigh = np.zeros((self.params['subj_data'] + self.params['input_features'] * num_bins, len(thigh_peak) - 1))
            
            # Load pre-trained OpenMetabolics' energy expenditure model
            self.data_driven_ee_model = pickle.load(open(self.params['model_weight'] + '/data_driven_ee_model.pkl', 'rb'))
            openMet_t_idx, openMet_time, openMet_ee, gait_data_thigh = self.estimateMetabolics(
                imu_time, self.data_driven_ee_model, gait_data_thigh, imu_gyro_acc, thigh_peak, stride_detect_window
            )

            # Pedometer estimation
            ped_time, ped_met = self.estimatePedometer(imu_time, thigh_peak, stride_detect_window, self.gender, self.body_weight)

            # Load heart rate data
            csv_file_path = os.path.join(val_data_path, cur_subj, 'hr_data.csv')
            df_hr_data = pd.read_csv(csv_file_path)
            hr_data = df_hr_data['hr_data (bpm)'].values
            hr_time = df_hr_data['time (s)'].values

            hr_met = self.estimateHeartrate(hr_data, self.body_weight, self.age, self.gender)

            # Load smartwatch data
            csv_file_path = os.path.join(val_data_path, cur_subj, 'smartwatch_est.csv')
            df_smartwatch = pd.read_csv(csv_file_path)
            sw_time = df_smartwatch['time (s)'].values
            sw_met = df_smartwatch['energy_estimates (W)'].values

            # Determine cumulative time range
            cum_time_i = cur_gt_met_time[0]
            cum_time_e = cur_gt_met_time[-1] - 180

            # Calculate cumulative metabolic cost (COSMED)
            cum_met = self.get_cum_met(cur_gt_met, cur_gt_met_time)
            cum_met = (cum_met - 180 * self.rest_met) / (cur_gt_met_time[-1] - cur_gt_met_time[0])

            # Calculate cumulative energy expenditure for OpenMetabolics
            # cum_time_openMet = imu_time[openMet_t_idx]
            cum_idx_openMet = (openMet_time > cum_time_i) * (openMet_time < cum_time_e)
            cum_time_openMet = openMet_time[cum_idx_openMet]
            cum_openMet = self.get_cum_met(openMet_ee, cum_time_openMet) / (cur_gt_met_time[-1] - cur_gt_met_time[0])

            # Calculate cumulative energy expenditure for Smartwatch
            cum_idx_sw = (sw_time > cum_time_i) * (sw_time < cum_time_e)
            cum_time_sw = sw_time[cum_idx_sw]
            cum_sw = self.get_cum_met(sw_met[cum_idx_sw], cum_time_sw) / (cur_gt_met_time[-1] - cur_gt_met_time[0])

            # Calculate cumulative energy expenditure for Heart rate model
            cum_idx_hr = (hr_time > cum_time_i) * (hr_time < cum_time_e)
            cum_time_hr = hr_time[cum_idx_hr]
            cum_hr = self.get_cum_met(hr_met[cum_idx_hr], cum_time_hr) / (cur_gt_met_time[-1] - cur_gt_met_time[0])

            # Calculate cumulative energy expenditure for Pedometer
            cum_idx_ped = (ped_time > cum_time_i) * (ped_time < cum_time_e)
            cum_time_ped = ped_time[cum_idx_ped]
            cum_ped = self.get_cum_met(ped_met[cum_idx_ped], cum_time_ped) / (cur_gt_met_time[-1] - cur_gt_met_time[0])

            # Calculate absolute error (%)
            abs_err_openmet = np.round(np.abs(cum_met - cum_openMet) / cum_met * 100, 3)[0]
            abs_err_sw_cum = np.round(np.abs(cum_met - cum_sw) / cum_met * 100, 3)[0]
            abs_err_hr_cum = np.round(np.abs(cum_met - cum_hr) / cum_met * 100, 3)[0]
            abs_err_ped_cum = np.round(np.abs(cum_met - cum_ped) / cum_met * 100, 3)[0]

            print("Absolute err (OpenMET cumulative):", abs_err_openmet)
            print("Absolute err (SmartWatch cumulative):", abs_err_sw_cum)
            print("Absolute err (HeartRate model cumulative):", abs_err_hr_cum)
            print("Absolute err (Pedometer cumulative):", abs_err_ped_cum)
            print("\n")

            abs_err_openmet_cum_list.append(abs_err_openmet)
            abs_err_sw_cum_list.append(abs_err_sw_cum)
            abs_err_hr_cum_list.append(abs_err_hr_cum)
            abs_err_ped_cum_list.append(abs_err_ped_cum)

            if self.params['plot_data']:
                fontsize = 7.5
                csfont = {'fontname': 'Arial'}
                title_fig = cur_subj
                fig, ax = plt.subplots(1, 1, figsize=(8, 2.5))
                ax.plot(cur_gt_met_time, cur_gt_met, '-', linewidth=1, color='black', label='Respirometry', alpha=1)
                ax.plot(openMet_time, openMet_ee, '-', linewidth=1, color='dodgerblue', label='OpenMetabolics', alpha=0.8)
                ax.plot(hr_time, hr_met, '-', linewidth=1, color='orangered', label='Heart-rate model', alpha=0.8)
                ax.plot(ped_time, ped_met, '-', linewidth=1, color='limegreen', label='Pedometer', alpha=0.8)
                ax.plot(sw_time, sw_met, '-', linewidth=1, color='violet', label='Smartwatch', alpha=0.8)
                ax.axvline(x=cum_time_i, color='grey', linestyle='--', label='Activity\nonset/offset', alpha=0.8)
                ax.axvline(x=cum_time_e, color='grey', linestyle='--', alpha=0.8)

                plt.xticks(fontsize=fontsize, **csfont)
                plt.yticks(fontsize=fontsize, **csfont)
                from matplotlib.font_manager import FontProperties
                font_properties = FontProperties(family='Arial', size=fontsize-3)
                ax.legend(frameon=False, prop=font_properties, loc='upper right')
                ax.set_ylabel('Energy expenditure (W)', fontsize=fontsize, **csfont)
                ax.set_xlabel('Absolute time(s)', fontsize=fontsize, **csfont)
                ax.spines['top'].set_color('none')
                ax.spines['right'].set_color('none')
                plt.suptitle(f'Real-world walking: {cur_subj}', fontsize=fontsize, **csfont)

                plt.tight_layout()
                save_path = './validation_results/indiv_plot/'
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(save_path + title_fig + '.png', dpi=300)

        # Convert lists to NumPy arrays for mean calculation
        abs_err_openmet_cum_list = np.array(abs_err_openmet_cum_list)
        abs_err_sw_cum_list = np.array(abs_err_sw_cum_list)
        abs_err_hr_cum_list = np.array(abs_err_hr_cum_list)
        abs_err_ped_cum_list = np.array(abs_err_ped_cum_list)

        # Calculate and print the mean of each MAPE
        mean_abs_err_openmet_cum = np.mean(abs_err_openmet_cum_list)
        mean_abs_err_sw_cum = np.mean(abs_err_sw_cum_list)
        mean_abs_err_hr_cum = np.mean(abs_err_hr_cum_list)
        mean_abs_err_ped_cum = np.mean(abs_err_ped_cum_list)

        print(f"Mean absolute error (OpenMET cumulative) : {mean_abs_err_openmet_cum:0.0f}%")
        print(f"Mean absolute error (SmartWatch cumulative) : {mean_abs_err_sw_cum:0.0f}%")
        print(f"Mean absolute error (HeartRate model cumulative): {mean_abs_err_hr_cum:0.0f}%")
        print(f"Mean absolute error (Pedometer cumulative) : {mean_abs_err_ped_cum:0.0f}%")

        if self.params['plot_data']:

            # Data and configurations for the box plot
            data = [abs_err_openmet_cum_list, abs_err_sw_cum_list, abs_err_ped_cum_list, abs_err_hr_cum_list]
            labels = ['OpenMetabolics', 'SmartWatch', 'Pedometer', 'Heart-rate model']
            colors = ['dodgerblue', 'orangered', 'limegreen', 'violet']
            means = [mean_abs_err_openmet_cum, mean_abs_err_sw_cum, mean_abs_err_ped_cum, mean_abs_err_hr_cum]

            bar_width = 0.3
            meanpointprops = dict(marker='s', markerfacecolor='black', markeredgecolor='black', markersize=3)
            fontsize = 7.5
            csfont = {'fontname': 'Arial'}
            fig_size = (4, 3)

            # Create the figure and axis
            fig, ax = plt.subplots()

            # Add the box plots with specific colors
            for i in range(len(data)):
                box = ax.boxplot(data[i], positions=[i + 1], patch_artist=True, widths=bar_width,
                                 boxprops=dict(facecolor=colors[i], color='black', alpha=0.8),
                                 meanprops=meanpointprops, medianprops=dict(color='k'),
                                 whiskerprops=dict(linestyle='-', linewidth=1.5),
                                 capprops=dict(color='black'), showfliers=False,
                                 flierprops=dict(marker='o', color='black', alpha=0.5),
                                 showmeans=True, notch=False)

                # Extract whisker values and calculate text position
                whisker_vals = [whisker.get_ydata()[1] for whisker in box['whiskers']]
                text_position = whisker_vals[1] + 2

                # Add mean values on top of the box plots
                ax.text(i + 1, text_position, f'{means[i]:.0f}%', fontsize=fontsize, ha='center', va='bottom', color='black', **csfont)

            # Set the x-ticks and labels
            ax.set_xticks([1, 2, 3, 4])
            ax.set_xticklabels(labels)

            # Set the title and labels
            ax.set_xlabel('Real-world\nwalking bouts\n$n$ = {}'.format(len(abs_err_openmet_cum_list)), fontsize=fontsize, **csfont)
            ax.set_ylabel('Absolute error (%)', fontsize=fontsize, **csfont)

            # Remove top and right spines
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')

            # Customize ticks
            plt.xticks(fontsize=fontsize, **csfont)
            plt.yticks(fontsize=fontsize, **csfont)

            # Adjust the figure size and layout
            fig.set_size_inches(fig_size)
            plt.tight_layout()

            # Save the figure
            save_path = './validation_results/'
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(save_path + 'real_world_walking_results.png', dpi=300)
