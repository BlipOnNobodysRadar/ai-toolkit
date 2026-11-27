from toolkit.paths import MODELS_PATH
import requests
import os
import json
import tqdm
import tempfile


class ModelCache:
    def __init__(self):
        self.raw_cache = {}
        self.cache_path = os.path.join(MODELS_PATH, '.ai_toolkit_cache.json')
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r') as f:
                all_cache = json.load(f)
                if 'models' in all_cache:
                    self.raw_cache = all_cache['models']
                else:
                    self.raw_cache = all_cache

    def get_model_path(self, model_id: int, model_version_id: int = None):
        if str(model_id) not in self.raw_cache:
            return None
        if model_version_id is None:
            # get latest version
            model_version_id = max([int(x) for x in self.raw_cache[str(model_id)].keys()])
            if model_version_id is None:
                return None
            model_path = self.raw_cache[str(model_id)][str(model_version_id)]['model_path']
            # check if model path exists
            if not os.path.exists(model_path):
                # remove version from cache
                del self.raw_cache[str(model_id)][str(model_version_id)]
                self.save()
                return None
            return model_path
        else:
            if str(model_version_id) not in self.raw_cache[str(model_id)]:
                return None
            model_path = self.raw_cache[str(model_id)][str(model_version_id)]['model_path']
            # check if model path exists
            if not os.path.exists(model_path):
                # remove version from cache
                del self.raw_cache[str(model_id)][str(model_version_id)]
                self.save()
                return None
            return model_path

    def update_cache(self, model_id: int, model_version_id: int, model_path: str):
        if str(model_id) not in self.raw_cache:
            self.raw_cache[str(model_id)] = {}
        if str(model_version_id) not in self.raw_cache[str(model_id)]:
            self.raw_cache[str(model_id)][str(model_version_id)] = {}
        self.raw_cache[str(model_id)][str(model_version_id)] = {
            'model_path': model_path
        }
        self.save()

    def save(self):
        if not os.path.exists(os.path.dirname(self.cache_path)):
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        all_cache = {'models': {}}
        if os.path.exists(self.cache_path):
            # load it first
            with open(self.cache_path, 'r') as f:
                all_cache = json.load(f)

        all_cache['models'] = self.raw_cache

        with open(self.cache_path, 'w') as f:
            json.dump(all_cache, f, indent=2)


def get_model_download_info(model_id: int, model_version_id: int = None):
    # curl https://civitai.com/api/v1/models?limit=3&types=TextualInversion \
    # -H "Content-Type: application/json" \
    # -X GET
    print(
        f"Getting model info for model id: {model_id}{f' and version id: {model_version_id}' if model_version_id is not None else ''}")
    endpoint = f"https://civitai.com/api/v1/models/{model_id}"

    # get the json
    response = requests.get(endpoint)
    response.raise_for_status()
    model_data = response.json()

    model_version = None

    # go through versions and get the top one if one is not set
    for version in model_data['modelVersions']:
        if model_version_id is not None:
            if str(version['id']) == str(model_version_id):
                model_version = version
                break
        else:
            # get first version
            model_version = version
            break

    if model_version is None:
        raise ValueError(
            f"Could not find a model version for model id: {model_id}{f' and version id: {model_version_id}' if model_version_id is not None else ''}")

    model_file = None
    # go through files and prefer fp16 safetensors
    # "metadata": {
    #   "fp": "fp16",
    #   "size": "pruned",
    #   "format": "SafeTensor"
    # },
    # todo check pickle scans and skip if not good
    # try to get fp16 safetensor
    for file in model_version['files']:
        if file['metadata']['fp'] == 'fp16' and file['metadata']['format'] == 'SafeTensor':
            model_file = file
            break

    if model_file is None:
        # try to get primary
        for file in model_version['files']:
            if file['primary']:
                model_file = file
                break

    if model_file is None:
        # try to get any safetensor
        for file in model_version['files']:
            if file['metadata']['format'] == 'SafeTensor':
                model_file = file
                break

    if model_file is None:
        # try to get any fp16
        for file in model_version['files']:
            if file['metadata']['fp'] == 'fp16':
                model_file = file
                break

    if model_file is None:
        # try to get any
        for file in model_version['files']:
            model_file = file
            break

    if model_file is None:
        raise ValueError(f"Could not find a model file to download for model id: {model_id}")

    return model_file, model_version['id']


def get_model_path_from_url(url: str):
    # get query params form url if they are set
    # https: // civitai.com / models / 25694?modelVersionId = 127742
    query_params = {}
    if '?' in url:
        query_string = url.split('?')[1]
        query_params = dict(qc.split("=") for qc in query_string.split("&"))

    # get model id from url
    model_id = url.split('/')[-1]
    # remove query params from model id
    if '?' in model_id:
        model_id = model_id.split('?')[0]
    if model_id.isdigit():
        model_id = int(model_id)
    else:
        raise ValueError(f"Invalid model id: {model_id}")

    model_cache = ModelCache()
    model_path = model_cache.get_model_path(model_id, query_params.get('modelVersionId', None))
    if model_path is not None:
        return model_path
    else:
        # download model
        file_info, model_version_id = get_model_download_info(model_id, query_params.get('modelVersionId', None))

        download_url = file_info['downloadUrl']  # url does not work directly
        filename = file_info['name']
        model_path = os.path.join(MODELS_PATH, filename)
        tmp_path = None
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"Downloading Civitai model {model_id}, version {model_version_id}")
        try:
            with requests.get(download_url, stream=True) as response:
                response.raise_for_status()
                total = int(response.headers.get('content-length', 0))
                with tempfile.NamedTemporaryFile(dir=os.path.dirname(model_path), prefix='.download_tmp_', delete=False) as f:
                    tmp_path = f.name
                    with tqdm.tqdm(total=total, unit='iB', unit_scale=True) as progress:
                        for data in response.iter_content(1024):
                            f.write(data)
                            progress.update(len(data))
            os.replace(tmp_path, model_path)
        except requests.RequestException as error:
            # HTTP and streaming errors can contain the full signed storage URL.
            status = f" (HTTP {error.response.status_code})" if error.response is not None else ""
            raise OSError(f"Civitai download failed for model {model_id}{status}") from None
        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                os.remove(tmp_path)
        model_cache.update_cache(model_id, model_version_id, model_path)
        return model_path


# if is main
if __name__ == '__main__':
    model_path = get_model_path_from_url("https://civitai.com/models/25694?modelVersionId=127742")
    print(model_path)
