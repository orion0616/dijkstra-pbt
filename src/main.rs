use proptest::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

fn main() {
    let source = 0;
    let graph = vec![
        vec![Edge { to: 1, weight: 1 }, Edge { to: 2, weight: 4 }],
        vec![Edge { to: 2, weight: 2 }, Edge { to: 3, weight: 5 }],
        vec![Edge { to: 3, weight: 1 }],
        vec![],
    ];
    let distances = dijkstra(&graph, source);
    println!("Distances from source {}: {:?}", source, distances);
}

/// グラフの各辺を表す構造体
#[derive(Clone, Debug)]
struct Edge {
    to: usize,
    weight: u32,
}

/// ダイクストラで使う状態（頂点とその現在のコスト）
#[derive(Copy, Clone, Eq, PartialEq)]
struct State {
    cost: u32,
    position: usize,
}

// BinaryHeap は最大ヒープなので、コストが小さいものを優先させるために順序を反転する
impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.cmp(&self.cost)
            .then_with(|| self.position.cmp(&other.position))
    }
}
impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// グラフは隣接リスト形式で表現
/// dijkstra は、与えられたソースから各頂点への最短距離を計算する。
/// 到達不可能な頂点は None となる。
fn dijkstra(graph: &Vec<Vec<Edge>>, source: usize) -> Vec<Option<u32>> {
    let n = graph.len();
    let mut dist = vec![None; n];
    let mut heap = BinaryHeap::new();

    // ソースの距離は 0
    dist[source] = Some(0);
    heap.push(State { cost: 0, position: source });

    while let Some(State { cost, position }) = heap.pop() {
        // すでにより短い経路が見つかっている場合はスキップ
        if Some(cost) != dist[position] {
            continue;
        }
        // 現在の頂点から出る各辺について緩和処理
        for edge in &graph[position] {
            let next_cost = cost + edge.weight;
            if dist[edge.to].is_none() || next_cost < dist[edge.to].unwrap() {
                dist[edge.to] = Some(next_cost);
                heap.push(State { cost: next_cost, position: edge.to });
            }
        }
    }
    dist
}

/// ランダムなグラフを生成するプロパティ（頂点数は 1～20）
/// 各頂点から出る辺は、ランダムな数（0～頂点数個）で、
/// 行き先は 0 〜 (num_vertices-1)、重みは 0～99 の範囲とする。
fn arb_graph() -> impl Strategy<Value = Vec<Vec<Edge>>> {
    (1usize..21).prop_flat_map(|num_vertices| {
        proptest::collection::vec(
            proptest::collection::vec((0..num_vertices, 0u32..100), 0..=num_vertices),
            num_vertices,
        )
        .prop_map(move |graph_tuples| {
            graph_tuples
                .into_iter()
                .map(|vec_tuples| {
                    vec_tuples
                        .into_iter()
                        .map(|(to, weight)| Edge { to, weight })
                        .collect::<Vec<Edge>>()
                })
                .collect::<Vec<Vec<Edge>>>()
        })
    })
}

/// プロパティテスト: 任意のグラフとソース頂点に対して、
/// 各辺 (u → v) について、
/// もし u と v の両方がソースから到達可能なら、
/// d(v) <= d(u) + weight(u,v) が成り立つはず。
proptest! {
    #[test]
    fn test_dijkstra_relaxation_property(graph in arb_graph(), source in any::<usize>()) {
        let n = graph.len();
        // ソース頂点を 0..n に収める
        let src = if n > 0 { source % n } else { 0 };

        println!("Testing graph: {:?}\nSource: {}", graph, src);

        let distances = dijkstra(&graph, src);

        for (u, edges) in graph.iter().enumerate() {
            // 頂点 u がソースから到達可能なら
            if let Some(d_u) = distances[u] {
                for edge in edges {
                    // 隣接頂点 v も到達可能ならチェック
                    if let Some(d_v) = distances[edge.to] {
                        prop_assert!(d_v <= d_u + edge.weight,
                            "Edge from {} to {} with weight {}: d({}) = {:?}, d({}) = {:?}",
                            u, edge.to, edge.weight, u, d_u, edge.to, d_v
                        );
                    }
                }
            }
        }
    }
}