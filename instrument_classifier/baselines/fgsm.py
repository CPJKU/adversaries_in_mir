import os
import torch
import torch.nn as nn
import torch.optim as optim

from instrument_classifier.utils.paths import log_path, d_path
from instrument_classifier.baselines.baseline_helper import prep_dir, prep_logger
from instrument_classifier.utils.utils import get_params, load_net, save_adv_example
from instrument_classifier.data.data_helper import get_raw_test_loader, make_get_feature
from instrument_classifier.utils.attack_utils import get_feature_fun, get_net, get_files, snr


def _fgsm(net, epsilon, x, y, get_feature):

    delta = torch.zeros_like(x).to(x.device)
    delta.requires_grad = True
    logits = net(get_feature(x + delta))  # same as if we'd directly work on x
    optimizer = optim.SGD([delta], lr=epsilon)  # lr corresponds to epsilon in original paper

    optimizer.zero_grad()
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    with torch.no_grad():
        delta.grad /= torch.abs(delta.grad)
        delta.grad[delta.grad != delta.grad] = 0.  # get rid of NANs
        delta.grad *= -1  # we want to add gradient
    optimizer.step()

    new_pred = net.predict(get_feature(x + delta)).item()
    orig_pred = net.predict(get_feature(x)).item()

    if orig_pred != new_pred:
        return (x + delta).view(1, -1), snr(x.detach().cpu().numpy(), (x + delta).detach().cpu().numpy()), 1
    return None, None, -1


def run_fgsm(net, data, get_feature, logger, ad_save_path, epsilon):
    """ Computes FGSM for all given data. """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()

    for i, (x, y) in enumerate(data):
        print('fgsm sample {}/{}'.format(i + 1, len(data)))

        x, y = x.to(device), y.to(device)
        file = data.dataset.filenames[i]
        file = file if '.wav' in file else file + '.wav'

        ad_ex, db, converged = _fgsm(net, epsilon, x.to(device), y.to(device), get_feature)
        orig_pred = net.predict(get_feature(x)).item()
        true_pred = y.item()
        if converged >= 0:
            print('\nsave current file: {} with distortion: {}db (r:{}/o:{})'.format(file, db, true_pred, orig_pred))
            new_pred = net.predict(get_feature(ad_ex)).item()
            save_adv_example(ad_ex.cpu(), os.path.join(ad_save_path, file))
            logger.append([file, converged, db, True, true_pred, orig_pred, new_pred])
        else:
            print('\ncould not find robust adv. example for file {}'.format(file))
            logger.append([file, converged, 0, False, true_pred, orig_pred, orig_pred])


def main():
    # get params
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    param_file = os.path.join(cur_dir, 'fgsm_config.txt')
    params = get_params(param_file)

    # load model to be attacked
    model_name = params.model_to_attack
    feature_fun = get_feature_fun(model_name)
    net = load_net(model_name, get_net())

    # prep directories for experiment
    ad_save_path = prep_dir(params.experiment)
    # prep data and feature computation
    files = sorted(get_files(params.valid))
    data_loader = get_raw_test_loader(files, d_path if params.valid else os.path.join(d_path, 'test'))
    get_feature = make_get_feature(feature_fun, os.path.join(log_path, model_name, '{}.csv'))
    # prep logger
    logger = prep_logger(param_file, params)

    # run fgsm
    run_fgsm(net, data_loader, get_feature, logger, ad_save_path, params.epsilon)


if __name__ == '__main__':
    main()
