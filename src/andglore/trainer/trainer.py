import time
from datetime import timedelta
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm

from andglore.evaluation.evaluate import adaptative_hac_evaluation
from andglore.model.andglore import ANDGloRe
from andglore.model.augmentation import (
    get_augmented_adjacencies,
    get_augmented_features,
)
from andglore.model.networks import create_bipartite_matrix, get_nodes
from andglore.trainer.logger import Logger
from andglore.trainer.training_args import TrainingArgs
from andglore.utils import AVERAGE_NAME, save_csv_results, set_seed


def run_andglore_experiment(
    dataset_name: str,
    networks_path,
    training_args: TrainingArgs,
    log_file_path: Optional[str] = None,
    results_csv_path: Optional[str] = None,
    outputs_path: Optional[str] = None,
    selected_names=None,
):
    # Logging
    logger = Logger(log_file_path).logger

    # Logging
    logger.info("Starting Experiment", extra={"show_time": True, "break_line": True})
    logger.info(f"Experiments Args:{training_args}", extra={"break_line": True})

    # Load the ambiguous networks
    networks = torch.load(
        networks_path,
        weights_only=False,
    )
    networks_by_name = {graph.graph["name"]: graph for graph in networks}

    # Prepare names
    names = selected_names
    if names is None:
        names = sorted(list(networks_by_name))
        # Logging
        logger.info(
            f"Running on all groups for dataset '{dataset_name}'",
            extra={"break_line": True},
        )

        # Logging
        logger.info(
            f"Running {len(training_args.seeds)} iterations with seeds: {training_args.seeds}",
            extra={"break_line": True},
        )

    # Training loop
    loop = tqdm(training_args.seeds, desc="AND-GloRe Experiment")

    # Track results across iterations and name groups
    results = {}
    outputs = {}

    for run, seed in enumerate(loop):

        results[seed] = {}
        outputs[seed] = {}

        live_pf1 = 0.0

        for name_index, name in enumerate(names):

            # Seed per name, so results are name deterministic
            set_seed(seed=seed)

            # Logging
            logger.info(
                f"Run {run}/{len(loop)} - Name {name_index}/{len(names)}: {name} - Avg. F1: {live_pf1:.4f}",
                extra={"show_time": True},
            )

            # Load the network
            network = networks_by_name[name]

            # Extract nodes
            ## Paper
            paper_nodes = get_nodes(network, "paper")
            node_count_paper = len(paper_nodes)
            ## Author
            author_nodes = get_nodes(network, "author")
            node_count_author = len(author_nodes)
            ## Venue
            venue_nodes = get_nodes(network, "venue")
            node_count_venue = len(venue_nodes)
            ## Org
            org_nodes = get_nodes(network, "org")
            node_count_org = len(org_nodes)

            # Extract paper features
            paper_features = (
                torch.stack([network.nodes[node]["feat"] for node in paper_nodes])
                .float()
                .cpu()
            )
            num_paper_features = paper_features.shape[1]

            # Extract paper labels
            paper_labels = torch.tensor(
                [network.nodes[node]["label"] for node in paper_nodes],
                dtype=torch.float32,
            )

            # Create bipartite matrices
            ## paper-author
            bipartite_pa = create_bipartite_matrix(
                network,
                "written_by",
                paper_nodes,
                author_nodes,
            )
            ## paper-venue
            bipartite_pv = create_bipartite_matrix(
                network,
                "published_in",
                paper_nodes,
                venue_nodes,
            )
            ## paper-org
            bipartite_po = create_bipartite_matrix(
                network,
                "author_affiliated_with",
                paper_nodes,
                org_nodes,
            )
            ## author-org
            bipartite_ao = create_bipartite_matrix(
                network,
                "affiliated_with",
                author_nodes,
                org_nodes,
            )

            # Create P-P metapaths adjacences
            ## P-A-P
            pp_adj_pap = (bipartite_pa @ bipartite_pa.T).tocoo()
            pp_adj_pap = torch.tensor(pp_adj_pap.toarray())
            pp_adj_pap.fill_diagonal_(1)
            ## P-V-P
            pp_adj_pvp = (bipartite_pv @ bipartite_pv.T).tocoo()
            pp_adj_pvp = torch.tensor(pp_adj_pvp.toarray())
            pp_adj_pvp.fill_diagonal_(1)
            ## P-O-P
            pp_adj_pop = (bipartite_po @ bipartite_po.T).tocoo()
            pp_adj_pop = torch.tensor(pp_adj_pop.toarray())
            pp_adj_pop.fill_diagonal_(1)
            ## P-A-O-A-P
            bipartite_pao = bipartite_pa @ bipartite_ao
            pp_adj_paoap = (bipartite_pao @ bipartite_pao.T).tocoo()  # type: ignore
            pp_adj_paoap = torch.tensor(pp_adj_paoap.toarray())
            pp_adj_paoap.fill_diagonal_(1)

            # Create global adjacency
            pp_adj_global = torch.stack(
                [pp_adj_pap, pp_adj_pvp, pp_adj_pop, pp_adj_paoap], dim=0
            ).mean(dim=0)

            # Features augmentation
            aug_paper_features = get_augmented_features(
                paper_features, feature_drop_rate=training_args.feat_mask
            )

            # P-P augmentation with one shared edge mask across metapaths
            aug_pp_adj_pap, aug_pp_adj_pvp, aug_pp_adj_pop, aug_pp_adj_paoap = (
                get_augmented_adjacencies(
                    [pp_adj_pap, pp_adj_pvp, pp_adj_pop, pp_adj_paoap],
                    edge_drop_rate=training_args.adj_mask,
                    hub_strength=1.0,
                )
            )

            # Init model
            model = ANDGloRe(
                num_papers=node_count_paper,
                num_authors=node_count_author,
                num_venues=node_count_venue,
                num_orgs=node_count_org,
                input_dim=num_paper_features,
                hidden_dim=training_args.hidden_dim,
                embedding_dim=training_args.embed_dim,
                projection_dim=training_args.proj_dim,
                network_encoder_hidden_dim=training_args.gnn_hidden_dim,
                temperature=training_args.temperature,
                dropout=training_args.dropout,
                network_encoder_dropout=training_args.gnn_dropout,
                network_encoder_alpha=training_args.gnn_alpha,
            ).to(training_args.device)

            # Set input to device
            paper_features = paper_features.to(training_args.device)
            pp_adj_global = pp_adj_global.to(training_args.device)
            pp_adj_pap = pp_adj_pap.to(training_args.device)
            pp_adj_pvp = pp_adj_pvp.to(training_args.device)
            pp_adj_pop = pp_adj_pop.to(training_args.device)
            pp_adj_paoap = pp_adj_paoap.to(training_args.device)
            aug_paper_features = aug_paper_features.to(training_args.device)
            aug_pp_adj_pap = aug_pp_adj_pap.to(training_args.device)
            aug_pp_adj_pvp = aug_pp_adj_pvp.to(training_args.device)
            aug_pp_adj_pop = aug_pp_adj_pop.to(training_args.device)
            aug_pp_adj_paoap = aug_pp_adj_paoap.to(training_args.device)

            # Convert bipartite to tensor
            bipartite_pa = torch.from_numpy(bipartite_pa.toarray()).to(training_args.device)  # type: ignore
            bipartite_pv = torch.from_numpy(bipartite_pv.toarray()).to(training_args.device)  # type: ignore
            bipartite_po = torch.from_numpy(bipartite_po.toarray()).to(training_args.device)  # type: ignore
            bipartite_ao = torch.from_numpy(bipartite_ao.toarray()).to(training_args.device)  # type: ignore
            bipartite_pao = torch.from_numpy(bipartite_pao.toarray()).to(training_args.device)  # type: ignore

            # Create optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=training_args.lr, weight_decay=training_args.l2_coef)  # type: ignore

            model.train()

            start_time = time.time()

            # Training loop
            try:
                for epoch in range(training_args.epochs):
                    optimizer.zero_grad()
                    loss = model(
                        paper_features,
                        bipartite_pa,
                        bipartite_pv,
                        bipartite_po,
                        bipartite_ao,
                        bipartite_pao,
                        pp_adj_global,
                        pp_adj_pap,
                        pp_adj_pvp,
                        pp_adj_pop,
                        pp_adj_paoap,
                        aug_paper_features,
                        aug_pp_adj_pap,
                        aug_pp_adj_pvp,
                        aug_pp_adj_pop,
                        aug_pp_adj_paoap,
                    )
                    loss.backward()
                    optimizer.step()
                    loop.set_postfix(
                        name=f"{name_index}/{len(names)}:{name}",
                        epoch=f"{epoch}/{training_args.epochs}",
                        loss=loss.item(),
                        run=f"{run+1}/{len(training_args.seeds)}",
                        seed=seed,
                        avg_pf1=f"{(live_pf1/(name_index+1)):.2%}",
                    )
            except Exception as e:
                # Logging
                logger.error(e, extra={"break_line": True})
                continue

            # Calculate network runtime
            runtime = int(time.time() - start_time)

            # Calculate evaluation metrics
            pred, score, distance, (pP, pR, pF1), (bP, bR, bF1), ari = (
                adaptative_hac_evaluation(
                    embeddings=model.refined_embeddings,  # type: ignore
                    paper_labels=paper_labels,
                    min_distance_threshold=training_args.min_distance_threshold,
                    max_distance_threshold=training_args.max_distance_threshold,
                    step=training_args.step,
                )
            )

            # Logging
            logger.info(f"Results for name: {name}", extra={"show_time": True})
            logger.info(f"Runtime: {timedelta(seconds=runtime)}")
            logger.info(f"pP: {pP:.2%} pR: {pR:.2%} pF1: {pF1:.2%}")
            logger.info(f"bP: {bP:.2%} bR: {bR:.2%} bF1: {bF1:.2%}")
            logger.info(f"ARI: {ari:.2%}")
            logger.info("")

            # Save results
            results[seed][name] = {
                "pP": pP,
                "pR": pR,
                "pF1": pF1,
                "bP": bP,
                "bR": bR,
                "bF1": bF1,
                "ari": ari,
                "runtime": runtime,
            }
            outputs[seed][name] = {
                "output": model.refined_embeddings,
                "pred": pred,
                "labels": paper_labels,
            }
            live_pf1 = live_pf1 + pF1
    # Save results
    results_df = save_csv_results(results, results_csv_path, outputs, outputs_path)

    # Logging
    overall = results_df[results_df["name"] == AVERAGE_NAME].iloc[0]
    logger.info(
        "Average Results Across All Names",
        extra={"show_time": True},
    )
    logger.info(
        f"Runtime: {timedelta(seconds=int(overall['runtime_mean']))} "
        f"(± {timedelta(seconds=int(overall['runtime_std']))})"
    )
    logger.info(
        f"pP: {overall['pP_mean']:.2%} (± {overall['pP_std']:.2%}) "
        f"pR: {overall['pR_mean']:.2%} (± {overall['pR_std']:.2%}) "
        f"pF1: {overall['pF1_mean']:.2%} (± {overall['pF1_std']:.2%})"
    )
    logger.info(
        f"bP: {overall['bP_mean']:.2%} (± {overall['bP_std']:.2%}) "
        f"bR: {overall['bR_mean']:.2%} (± {overall['bR_std']:.2%}) "
        f"bF1: {overall['bF1_mean']:.2%} (± {overall['bF1_std']:.2%})"
    )
    logger.info(f"ARI: {overall['ari_mean']:.2%} (± {overall['ari_std']:.2%})")
    logger.info("")
