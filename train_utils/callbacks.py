import keras, os, time, json, requests
# Callback to save metrics to a file
class SaveCustomMetrics(keras.callbacks.Callback):
    def __init__(self, output_folder):
        self.output_folder=output_folder
        
    def on_epoch_end(self, epoch, logs):
        data = dict()
        with open(os.path.join(self.output_folder, 'custom_logs.json'), 'r') as of:
            data=json.load(of)
            
        data[epoch]=logs
        with open(os.path.join(self.output_folder, 'custom_logs.json'), 'w') as of:
            json.dump(data, of)


# Save per batch training times
class BatchTimer(keras.callbacks.Callback):
    def __init__(self, output_folder):
        self.output_folder=output_folder
        
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch=epoch
        
    def on_batch_begin(self, batch, logs={}):
        self.batch_time_start = time.time()

    def on_batch_end(self, batch, logs={}):
        output_file = os.path.join(self.output_folder, 'batch_times.txt')
        batch_time = time.time() - self.batch_time_start
        with open(output_file, 'at') as of:
            of.write('{epoch} {batch} {time}\n'.format(epoch=self.epoch, batch=batch, time=batch_time))
            
   
# Send slack message
class SlackCallback(keras.callbacks.Callback):
    def __init__(self, URLs, start_msg, stop_msg):
        self.URLs = URLs
        self.start_msg = start_msg
        self.stop_msg = stop_msg
        
    def on_train_begin(self, logs={}):
        for url in self.URLs:
            r = requests.post(url, json={'text': self.start_msg})
            print(r.status_code, r.reason)

    def on_train_end(self, logs={}):
        for url in self.URLs:
            r = requests.post(url, json={'text': self.stop_msg})
            print(r.status_code, r.reason)
