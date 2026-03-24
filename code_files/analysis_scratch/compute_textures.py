from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds Han_AIR/ to path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # adds Han_AIR/ to path
import code_files.analysis_scratch.analysis_utils as au
import matplotlib.pyplot as plt




def calc_features(name,sub_d,params):
    """making it also return for the sake of the multiprocessing.
    params is like: {'window':(16, 16), 'step':(16, 16)}"""
    input_map_name = 'basic_slab_avg_map|[5, 25]'

    big_key = f"{input_map_name}__{params}" 
    if big_key in au.ALL_EXISTING_MAPS:
        print(f"will be skipping {big_key}")
        return None,None
    print(f"will be processing {big_key}")
    glcm_features,_ = au.glcm_feature_maps(sub_d['maps'][input_map_name],**params)
    glcm_features = {f"{big_key}__{k}":v for k,v in glcm_features.items()}
    # sub_d['maps'].update(glcm_features) 
    return name, glcm_features

if __name__ == "__main__":
    import pickle
    # GLOBAL_RECOMPUTE=False
    constants_dict = {'OVERWRITE_FEATURES':False,
                      'REGISTER_MAPS':True,}
    print(f'running script with constants dict as {constants_dict}')


    if constants_dict['OVERWRITE_FEATURES']:
        print("entering feature overwrite workflow")
        print("loading all basic maps")
        maps_dict = pickle.load(open('/Users/matthewhunt/Research/Iowa_Research/Han_AIR/results/all_maps_dict.pickle','rb'))

        print("now calculating texture based features on those maps")
        maps_to_use = ['basic_slab_avg_map|[5, 25]']
        for name,sub_d in maps_dict.items():
            for k in list(sub_d['maps'].keys()):
                if k not in maps_to_use:
                    # print(f'deleting k: {k}')
                    del sub_d['maps'][k]


        print(f"goign to use the dict {maps_dict.keys()}")

        param_combos = [
            {'window':(32, 32), 'step':(4,4)},
        ]
        from concurrent.futures import ProcessPoolExecutor
        glcm_feature_dict = {}
        DEBUG = False
        if DEBUG:
            futures = []
            for name,sub_d in maps_dict.items():
                for params in param_combos:
                    futures.append(calc_features(name, sub_d,params))
        else:
            with ProcessPoolExecutor(max_workers=12) as exe:
                futures = [
                    exe.submit(calc_features, name, sub_d,params)
                    for name,sub_d in maps_dict.items()
                    for params in param_combos
                ]
                # collect results and sort by bscan_idx
                futures = [fut.result() for fut in futures]

        

        for name, updated_sub_d in futures:
            print(f"now updating {name}")
            if name is None: # If already existed
                continue
            if name not in glcm_feature_dict:
                glcm_feature_dict[name] = {'maps':{}} # Will also include all the original results
            glcm_feature_dict[name]['maps'].update(updated_sub_d)

        k0 = next(iter(glcm_feature_dict))
        all_features = glcm_feature_dict[k0]['maps'].keys()
        print(f"plotting all features: {all_features}")
        for feat_key in all_features:
            au.plot_maps(glcm_feature_dict,feat_key)
        pickle.dump(glcm_feature_dict,open('/Users/matthewhunt/Research/Iowa_Research/Han_AIR/results/all_features_dict.pickle','wb'))
    else:
        glcm_feature_dict = pickle.load(open('/Users/matthewhunt/Research/Iowa_Research/Han_AIR/results/all_features_dict.pickle','rb'))
    if constants_dict['REGISTER_MAPS']:
        print("\nnow working to register the maps")

        def quickplot(img,points=None,downsampling=1):
            plt.figure()
            plt.imshow(img,cmap='gray',aspect='auto')
            if points:
                ys, xs = zip(*points)
                xs = [e/downsampling for e in xs]
                ys = [e/downsampling for e in ys]
                plt.scatter(xs, ys, s=20, marker='x')
            plt.show()

        k0 = next(iter(glcm_feature_dict))
        print(k0)
        labels_root = "/Users/matthewhunt/Research/Iowa_Research/Han_AIR/testing_annotations"
        suffix = '.labels.zarr'
        label_path = (Path(labels_root) / k0).with_suffix(suffix)

        all_features = glcm_feature_dict[k0]['maps'].keys()
        map = glcm_feature_dict[k0]['maps'][list(all_features)[0]]

        
        print('calculating centroids')
        centroids = au.get_centroids_from_annotations(label_path)
        centroids = {k:[e/4 for e in v] for k,v in centroids.items()}
        # quickplot(glcm_feature_dict[k0]['maps'][next(iter(all_features))])

        print('plotting centroids')
        # quickplot(map,points = list(centroids.values()),downsampling=4)

        out = au.align_by_fovea_onh(map,centroids['fovea_center'],centroids['onh_center'])
        affine_map = out[0]


        quickplot(affine_map)
        import pdb; pdb.set_trace()


        