export const API_BASE_URL = '/api';

export interface Comment {
    comment_id: string;
    parent_id: string;
    Source: string;
    Author: string;
    Text: string;
    Score: number;
    Date: string;
    Word_Count: number;
}

export interface Item {
    ID: string;
    Source: string;
    Type: string;
    Author: string;
    Title: string | null;
    Text: string;
    Score?: number;
    Date: string;
    Word_Count?: number;
    Comments?: Comment[];
}

export interface Stats {
    labeled_count: number;
    remaining_count: number;
    total_assigned: number;
}

export async function verifySecret(secret_key: string): Promise<boolean> {
    try {
        const response = await fetch(`${API_BASE_URL}/auth/verify`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ secret_key }),
        });
        if (!response.ok) return false;
        const data = await response.json();
        return data.valid === true;
    } catch (err) {
        console.error('Failed to verify secret:', err);
        return false;
    }
}

export async function getLabelers(): Promise<string[]> {
    try {
        const response = await fetch(`${API_BASE_URL}/labelers`);
        if (!response.ok) throw new Error('Failed to fetch labelers');
        return response.json();
    } catch (err) {
        console.error('Failed to get labelers:', err);
        return [];
    }
}

export async function getNextItem(labelerName: string): Promise<Item | null> {
    try {
        const response = await fetch(`${API_BASE_URL}/labelers/${labelerName}/items/next`);
        if (response.status === 404) return null; // No more items
        if (!response.ok) throw new Error('Failed to fetch next item');
        return response.json();
    } catch (err) {
        console.error('Failed to get next item:', err);
        return null;
    }
}

export async function getStats(labelerName: string): Promise<Stats | null> {
    try {
        const response = await fetch(`${API_BASE_URL}/labelers/${labelerName}/stats`);
        if (!response.ok) throw new Error('Failed to fetch stats');
        return response.json();
    } catch (err) {
        console.error('Failed to get stats:', err);
        return null;
    }
}

export async function recordLabel(
    labelerName: string, 
    itemId: string, 
    label: 'positive' | 'negative' | 'neutral' | 'irrelevant',
    commentLabels: Record<string, string> = {}
): Promise<boolean> {
    try {
        const response = await fetch(`${API_BASE_URL}/labels`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                labeler_name: labelerName,
                item_id: itemId,
                label,
                comment_labels: commentLabels
            }),
        });
        return response.ok;
    } catch (err) {
        console.error('Failed to record label:', err);
        return false;
    }
}
