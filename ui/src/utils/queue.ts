import { apiClient } from '@/utils/api';

export const startQueue = async (queueID: string): Promise<void> => {
  await apiClient.get(`/api/queue/${queueID}/start`);
};

export const stopQueue = async (queueID: string): Promise<void> => {
  await apiClient.get(`/api/queue/${queueID}/stop`);
};
