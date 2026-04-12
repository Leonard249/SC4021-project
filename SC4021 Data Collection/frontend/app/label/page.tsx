'use client';

import { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { getNextItem, getStats, recordLabel, Item, Stats } from '@/app/lib/api';

export default function LabelPage() {
    const router = useRouter();
    const [labelerName, setLabelerName] = useState<string | null>(null);
    const [item, setItem] = useState<Item | null>(null);
    const [stats, setStats] = useState<Stats | null>(null);
    const [loading, setLoading] = useState(true);
    const [submitting, setSubmitting] = useState(false);
    const [error, setError] = useState('');
    const [isDone, setIsDone] = useState(false);
    
    // Label state tracking
    const [mainLabel, setMainLabel] = useState<'positive' | 'negative' | 'neutral' | 'irrelevant' | null>(null);
    const [commentLabels, setCommentLabels] = useState<Record<string, 'positive' | 'negative' | 'neutral' | 'irrelevant'>>({});

    // Initialize and load first item
    const loadData = useCallback(async (name: string) => {
        try {
            setLoading(true);
            setError('');

            const [fetchedStats, fetchedItem] = await Promise.all([
                getStats(name),
                getNextItem(name)
            ]);

            setStats(fetchedStats);

            if (!fetchedItem) {
                setIsDone(true);
            } else {
                setItem(fetchedItem);
                setIsDone(false);
                // Reset labels
                setMainLabel(null);
                setCommentLabels({});
            }
        } catch (err) {
            setError('Failed to fetch data. Please try again.');
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        if (typeof window !== 'undefined') {
            const storedName = localStorage.getItem('labeler_name');
            const secret = localStorage.getItem('auth_secret');

            if (!secret || !storedName) {
                router.push('/');
                return;
            }
            setLabelerName(storedName);
            loadData(storedName);
        }
    }, [router, loadData]);

    const handleSubmit = async () => {
        if (!labelerName || !item || submitting || !mainLabel) return;

        // Validation for comments
        const requiredCommentCount = item.Comments ? item.Comments.length : 0;
        const currentCommentLabelsCount = Object.keys(commentLabels).length;
        if (currentCommentLabelsCount !== requiredCommentCount) {
             setError('Please label all comments before submitting.');
             return;
        }

        setSubmitting(true);
        try {
            const success = await recordLabel(labelerName, item.ID, mainLabel, commentLabels);
            if (success) {
                await loadData(labelerName);
            } else {
                setError('Failed to submit label. Please try again.');
            }
        } catch (err) {
            setError('An error occurred submitting the label.');
        } finally {
            setSubmitting(false);
        }
    };

    const isSubmitReady = () => {
        if (!mainLabel) return false;
        if (item?.Comments && Object.keys(commentLabels).length !== item.Comments.length) return false;
        return true;
    };

    const handleLogout = () => {
        if (typeof window !== 'undefined') {
            localStorage.removeItem('labeler_name');
        }
        router.push('/labeler');
    };

    if (loading && !item) {
        return (
            <main className="flex min-h-screen items-center justify-center bg-slate-50">
                <div className="flex animate-pulse flex-col items-center gap-4">
                    <div className="h-12 w-12 rounded-full border-4 border-indigo-600 border-t-transparent animate-spin"></div>
                    <p className="text-indigo-900 font-medium">Loading next document...</p>
                </div>
            </main>
        );
    }

    if (isDone) {
        return (
            <main className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-br from-indigo-50 to-purple-50 p-4">
                <div className="w-full max-w-lg rounded-3xl bg-white p-10 text-center shadow-xl border border-indigo-100">
                    <div className="mx-auto mb-6 flex h-24 w-24 items-center justify-center rounded-full bg-green-100 text-green-500">
                        <svg className="h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                        </svg>
                    </div>
                    <h1 className="mb-2 text-3xl font-extrabold text-gray-900">All Done!</h1>
                    <p className="mb-8 text-gray-500">You have completed all your assigned labeling tasks for today. Great job, <span className="capitalize font-semibold">{labelerName}</span>!</p>
                    <button
                        onClick={handleLogout}
                        className="rounded-xl bg-indigo-600 px-6 py-3 font-semibold text-white shadow-md hover:bg-indigo-700 transition"
                    >
                        Back to Profiles
                    </button>
                </div>
            </main>
        );
    }

    // Progress percentage
    const progressPercent = stats && stats.total_assigned > 0
        ? Math.round((stats.labeled_count / stats.total_assigned) * 100)
        : 0;

    return (
        <main className="flex min-h-screen flex-col bg-slate-50 text-slate-800">
            {/* Header / Stats */}
            <header className="sticky top-0 z-10 bg-white/80 backdrop-blur-md border-b border-slate-200 px-4 py-3 shadow-sm">
                <div className="mx-auto flex max-w-3xl items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 font-bold text-white shadow-sm">
                            {labelerName?.charAt(0).toUpperCase()}
                        </div>
                        <div>
                            <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">Labeler</p>
                            <p className="font-semibold capitalize text-slate-900 leading-tight">{labelerName}</p>
                        </div>
                    </div>

                    <div className="text-right">
                        <p className="text-sm font-semibold text-indigo-600">
                            {progressPercent}% Complete
                        </p>
                        <p className="text-xs text-slate-500">
                            {stats?.labeled_count} / {stats?.total_assigned}
                        </p>
                    </div>
                </div>

                {/* Progress Bar */}
                <div className="mx-auto mt-3 h-2 max-w-3xl overflow-hidden rounded-full bg-slate-100">
                    <div
                        className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 transition-all duration-500 ease-out"
                        style={{ width: `${progressPercent}%` }}
                    />
                </div>
            </header>

            {error && (
                <div className="mx-auto mt-4 w-full max-w-3xl px-4">
                    <div className="rounded-xl bg-red-50 p-4 text-sm text-red-600 border border-red-200">
                        {error}
                    </div>
                </div>
            )}

            {/* Main Content Area */}
            {item && (
                <div className="mx-auto w-full max-w-3xl flex-1 px-4 py-6 md:py-8 flex flex-col">
                    <article className="flex-1 rounded-3xl bg-white p-6 md:p-8 shadow-sm ring-1 ring-slate-200/50 flex flex-col">
                        <div className="mb-6 flex flex-wrap items-center gap-2 text-xs font-medium text-slate-500">
                            <span className="rounded-full bg-slate-100 px-3 py-1 font-mono text-[10px] sm:text-xs">ID: {item.ID.length > 20 ? item.ID.substring(0, 20) + '...' : item.ID}</span>
                            {item.Source && <span className="rounded-full bg-indigo-50 text-indigo-700 px-3 py-1">{item.Source}</span>}
                            {item.Type && <span className="rounded-full bg-purple-50 text-purple-700 px-3 py-1">{item.Type}</span>}
                            {item.Date && <span className="px-1">{item.Date}</span>}
                        </div>

                        <h2 className="mb-4 text-2xl md:text-3xl font-bold tracking-tight text-slate-900 leading-snug">
                            {item.Title || "No Title"}
                        </h2>

                        {item.Author && (
                            <p className="mb-6 text-sm text-slate-500">By {item.Author}</p>
                        )}

                        <div className="prose prose-slate prose-lg max-w-none text-slate-700 prose-p:leading-relaxed flex-1">
                            {item.Text.split('\n').map((paragraph, i) => (
                                <p key={i} className="mb-4">{paragraph}</p>
                            ))}
                        </div>

                        {/* Comments Section */}
                        {item.Comments && item.Comments.length > 0 && (
                            <div className="mt-8 border-t border-slate-200 pt-6">
                                <h3 className="mb-4 text-xl font-bold text-slate-800 flex items-center gap-2">
                                    <svg className="w-5 h-5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" /></svg>
                                    Comments ({item.Comments.length})
                                </h3>
                                <div className="space-y-4">
                                    {item.Comments.map((comment) => (
                                        <div key={comment.comment_id} className="rounded-xl bg-slate-50 p-4 ring-1 ring-slate-200/60">
                                            <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500">
                                                <div className="flex flex-wrap items-center gap-2">
                                                    <span className="font-semibold text-slate-700">{comment.Author}</span>
                                                    {comment.Score !== undefined && (
                                                        <span className="flex items-center gap-1 rounded bg-slate-200 px-1.5 py-0.5 font-mono text-[10px] text-slate-600">
                                                            ↑ {comment.Score}
                                                        </span>
                                                    )}
                                                </div>
                                                <span className="text-[10px]">{comment.Date}</span>
                                            </div>
                                            <div className="prose prose-sm prose-slate max-w-none prose-p:leading-relaxed mb-3">
                                                {comment.Text.split('\n').map((paragraph, i) => (
                                                    <p key={i} className="mb-2 last:mb-0">{paragraph}</p>
                                                ))}
                                            </div>

                                            {/* Comment Label Buttons */}
                                            <div className="flex flex-wrap gap-2 mt-2 pt-3 border-t border-slate-200">
                                                <button
                                                    onClick={() => setCommentLabels(prev => ({...prev, [comment.comment_id]: 'positive'}))}
                                                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1 transition-all
                                                        ${commentLabels[comment.comment_id] === 'positive' 
                                                            ? 'bg-emerald-500 text-white shadow-sm ring-1 ring-emerald-600' 
                                                            : 'bg-white text-slate-600 hover:bg-emerald-50 ring-1 ring-slate-200'}`}
                                                >
                                                    <span>👍</span> {commentLabels[comment.comment_id] === 'positive' ? 'Selected' : 'Positive'}
                                                </button>
                                                <button
                                                    onClick={() => setCommentLabels(prev => ({...prev, [comment.comment_id]: 'neutral'}))}
                                                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1 transition-all
                                                        ${commentLabels[comment.comment_id] === 'neutral' 
                                                            ? 'bg-slate-600 text-white shadow-sm ring-1 ring-slate-700' 
                                                            : 'bg-white text-slate-600 hover:bg-slate-100 ring-1 ring-slate-200'}`}
                                                >
                                                    <span>😐</span> {commentLabels[comment.comment_id] === 'neutral' ? 'Selected' : 'Neutral'}
                                                </button>
                                                <button
                                                    onClick={() => setCommentLabels(prev => ({...prev, [comment.comment_id]: 'negative'}))}
                                                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1 transition-all
                                                        ${commentLabels[comment.comment_id] === 'negative' 
                                                            ? 'bg-rose-500 text-white shadow-sm ring-1 ring-rose-600' 
                                                            : 'bg-white text-slate-600 hover:bg-rose-50 ring-1 ring-slate-200'}`}
                                                >
                                                    <span>👎</span> {commentLabels[comment.comment_id] === 'negative' ? 'Selected' : 'Negative'}
                                                </button>
                                                <button
                                                    onClick={() => setCommentLabels(prev => ({...prev, [comment.comment_id]: 'irrelevant'}))}
                                                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1 transition-all
                                                        ${commentLabels[comment.comment_id] === 'irrelevant' 
                                                            ? 'bg-amber-500 text-white shadow-sm ring-1 ring-amber-600' 
                                                            : 'bg-white text-slate-600 hover:bg-amber-50 ring-1 ring-slate-200'}`}
                                                >
                                                    <span>🚫</span> {commentLabels[comment.comment_id] === 'irrelevant' ? 'Selected' : 'Irrelevant'}
                                                </button>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </article>

                    {/* Main Action Buttons */}
                    <div className="sticky bottom-4 mt-6 md:static md:bottom-auto flex flex-col gap-4">
                        <div>
                            <p className="text-sm font-semibold text-slate-600 mb-2 px-1">Label for Overall Content</p>
                            <div className="grid grid-cols-2 gap-3 rounded-2xl bg-white/90 p-3 shadow-sm backdrop-blur ring-1 ring-slate-200 md:grid-cols-4 md:bg-transparent md:p-0 md:shadow-none md:ring-0 md:backdrop-blur-none">
                                <button
                                    onClick={() => setMainLabel('positive')}
                                    className={`group flex flex-col items-center justify-center rounded-xl py-3 transition-all active:scale-[0.98] ring-1
                                        ${mainLabel === 'positive' 
                                            ? 'bg-emerald-500 text-white shadow-md ring-emerald-600' 
                                            : 'bg-emerald-50 text-emerald-700 hover:bg-emerald-100 ring-emerald-200/50 hover:ring-transparent'}`}
                                >
                                    <span className="text-xl mb-1 group-hover:scale-110 transition-transform">👍</span>
                                    <span className="text-xs font-bold uppercase tracking-wide">Positive</span>
                                </button>

                                <button
                                    onClick={() => setMainLabel('neutral')}
                                    className={`group flex flex-col items-center justify-center rounded-xl py-3 transition-all active:scale-[0.98] ring-1
                                        ${mainLabel === 'neutral' 
                                            ? 'bg-slate-600 text-white shadow-md ring-slate-700' 
                                            : 'bg-slate-50 text-slate-600 hover:bg-slate-200 ring-slate-200/60 hover:ring-transparent'}`}
                                >
                                    <span className="text-xl mb-1 group-hover:scale-110 transition-transform">😐</span>
                                    <span className="text-xs font-bold uppercase tracking-wide">Neutral</span>
                                </button>

                                <button
                                    onClick={() => setMainLabel('negative')}
                                    className={`group flex flex-col items-center justify-center rounded-xl py-3 transition-all active:scale-[0.98] ring-1
                                        ${mainLabel === 'negative' 
                                            ? 'bg-rose-500 text-white shadow-md ring-rose-600' 
                                            : 'bg-rose-50 text-rose-700 hover:bg-rose-100 ring-rose-200/50 hover:ring-transparent'}`}
                                >
                                    <span className="text-xl mb-1 group-hover:scale-110 transition-transform">👎</span>
                                    <span className="text-xs font-bold uppercase tracking-wide">Negative</span>
                                </button>

                                <button
                                    onClick={() => setMainLabel('irrelevant')}
                                    className={`group flex flex-col items-center justify-center rounded-xl py-3 transition-all active:scale-[0.98] ring-1
                                        ${mainLabel === 'irrelevant' 
                                            ? 'bg-amber-500 text-white shadow-md ring-amber-600' 
                                            : 'bg-amber-50 text-amber-700 hover:bg-amber-100 ring-amber-200/50 hover:ring-transparent'}`}
                                >
                                    <span className="text-xl mb-1 group-hover:scale-110 transition-transform">🚫</span>
                                    <span className="text-xs font-bold uppercase tracking-wide">Irrelevant</span>
                                </button>
                            </div>
                        </div>

                        {/* Submit Final Button */}
                        <div className="md:mt-4">
                            <button
                                disabled={!isSubmitReady() || submitting}
                                onClick={handleSubmit}
                                className="w-full rounded-xl bg-indigo-600 px-6 py-4 font-bold text-white shadow-lg transition-all hover:bg-indigo-700 focus:outline-none focus:ring-4 focus:ring-indigo-500/30 disabled:opacity-50 disabled:pointer-events-none active:scale-[0.98] flex items-center justify-center gap-2"
                            >
                                {submitting ? (
                                    <>
                                        <div className="h-5 w-5 rounded-full border-2 border-white border-t-transparent animate-spin"></div>
                                        Submitting...
                                    </>
                                ) : (
                                    <>
                                        Submit & Next <span className="text-lg">→</span>
                                    </>
                                )}
                            </button>
                            {(!isSubmitReady() && mainLabel) && (
                                <p className="text-center text-xs text-slate-500 mt-2">
                                    Please ensure you have labelled all {item.Comments?.length} comments before submitting.
                                </p>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </main>
    );
}
